"""Split-conformal operator error sets and empirical local Lipschitz estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


def _higher_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(values) + 1) * coverage)) - 1
    return float(values[min(max(rank, 0), len(values) - 1)])


@dataclass(frozen=True)
class OperatorCalibrator:
    multiplier: float
    coverage: float
    lipschitz: float
    norm_kind: str = "l2"

    def functional_norm(
        self,
        value: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_kind == "l2":
            return torch.sqrt(torch.mean(value.square(), dim=1))
        if self.norm_kind == "decision":
            x = torch.linspace(
                0.0,
                1.0,
                value.shape[1],
                dtype=value.dtype,
                device=value.device,
            )
            weight = 0.15 + 0.85 * torch.exp(-0.5 * ((x - 0.72) / 0.14).square())
            return torch.sqrt(torch.mean(weight[None, :] * value.square(), dim=1))
        if self.norm_kind != "ellipsoid":
            raise ValueError(f"Unknown norm: {self.norm_kind}")
        raise ValueError("Use standardized_score() for ellipsoidal set membership.")

    def standardized_score(
        self,
        error: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if self.norm_kind != "ellipsoid":
            return self.functional_norm(error) / self.functional_norm(scale).clamp_min(1e-7)
        return torch.sqrt(torch.mean((error / scale.clamp_min(1e-5)).square(), dim=1))

    def radius(
        self,
        scale: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a function-space radius or downstream support value."""

        if self.norm_kind != "ellipsoid":
            return self.multiplier * self.functional_norm(scale)
        if reference is None:
            raise ValueError("Ellipsoid support requires a cost sensitivity.")
        # Support function of a diagonal ellipsoid in the cost-gradient direction.
        return self.multiplier * torch.sqrt(torch.mean((reference * scale).square(), dim=1))

    def propagate(self, previous_radius: torch.Tensor, one_step_radius: torch.Tensor) -> torch.Tensor:
        return self.lipschitz * previous_radius + one_step_radius


@torch.no_grad()
def calibrate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    coverage: float = 0.90,
    perturbation: float = 0.01,
    norm_kind: str = "l2",
) -> tuple[OperatorCalibrator, dict[str, float]]:
    """Calibrate normalized residuals and estimate a local Lipschitz quantile."""

    model.eval()
    norm = OperatorCalibrator(1.0, coverage, 1.0, norm_kind)
    scores: list[np.ndarray] = []
    lipschitz_ratios: list[np.ndarray] = []
    for state, action, viscosity, boundary, target in loader:
        state = state.to(device)
        action = action.to(device)
        viscosity = viscosity.to(device)
        boundary = boundary.to(device)
        target = target.to(device)
        mean, scale = model(state, action, viscosity, boundary)
        scores.append(norm.standardized_score(target - mean, scale).cpu().numpy())

        noise = perturbation * torch.randn_like(state)
        noise[:, 0] = noise[:, -1] = 0.0
        perturbed_mean, _ = model(state + noise, action, viscosity, boundary)
        if norm_kind == "ellipsoid":
            numerator = torch.sqrt(torch.mean((perturbed_mean - mean).square(), dim=1))
            denominator = torch.sqrt(torch.mean(noise.square(), dim=1)).clamp_min(1e-7)
        else:
            numerator = norm.functional_norm(perturbed_mean - mean)
            denominator = norm.functional_norm(noise).clamp_min(1e-7)
        lipschitz_ratios.append((numerator / denominator).cpu().numpy())

    all_scores = np.concatenate(scores)
    all_lipschitz = np.concatenate(lipschitz_ratios)
    multiplier = _higher_quantile(all_scores, coverage)
    # A high but not maximum quantile avoids a single numerical outlier exploding the tube.
    lipschitz = max(1.0, float(np.quantile(all_lipschitz, 0.95)))
    calibrator = OperatorCalibrator(multiplier, coverage, lipschitz, norm_kind)
    metrics = {
        "multiplier": multiplier,
        "lipschitz_q95": lipschitz,
        "calibration_size": float(all_scores.size),
        "norm_kind": norm_kind,
    }
    return calibrator, metrics


@torch.no_grad()
def coverage_metrics(
    model: torch.nn.Module,
    calibrator: OperatorCalibrator,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    errors, radii = [], []
    for state, action, viscosity, boundary, target in loader:
        state, action = state.to(device), action.to(device)
        viscosity, boundary = viscosity.to(device), boundary.to(device)
        target = target.to(device)
        mean, scale = model(state, action, viscosity, boundary)
        if calibrator.norm_kind == "ellipsoid":
            x = torch.linspace(0.0, 1.0, mean.shape[1], dtype=mean.dtype, device=mean.device)
            weight = 0.15 + 0.85 * torch.exp(-0.5 * ((x - 0.72) / 0.14).square())
            reference = 2.0 * weight[None, :] * mean
            errors.append(
                torch.abs(torch.mean(reference * (target - mean), dim=1)).cpu().numpy()
            )
            radii.append(calibrator.radius(scale, reference).cpu().numpy())
            continue
        else:
            reference = mean
        errors.append(calibrator.functional_norm(target - mean, reference).cpu().numpy())
        radii.append(calibrator.radius(scale, reference).cpu().numpy())
    error = np.concatenate(errors)
    radius = np.concatenate(radii)
    if calibrator.norm_kind == "ellipsoid":
        membership_scores = []
        for state, action, viscosity, boundary, target in loader:
            state, action = state.to(device), action.to(device)
            viscosity, boundary = viscosity.to(device), boundary.to(device)
            target = target.to(device)
            mean, scale = model(state, action, viscosity, boundary)
            membership_scores.append(
                calibrator.standardized_score(target - mean, scale).cpu().numpy()
            )
        coverage = float(np.mean(np.concatenate(membership_scores) <= calibrator.multiplier))
    else:
        coverage = float(np.mean(error <= radius))
    return {
        "mean_functional_error": float(np.mean(error)),
        "coverage": coverage,
        "mean_radius": float(np.mean(radius)),
        "radius_error_correlation": float(np.corrcoef(radius, error)[0, 1]),
    }
