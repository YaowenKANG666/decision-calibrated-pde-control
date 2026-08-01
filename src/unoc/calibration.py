"""Split-conformal operator error sets and empirical local Lipschitz estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


def _higher_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0:
        raise ValueError("conformal calibration requires at least one score")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between zero and one")
    rank = int(np.ceil((len(values) + 1) * coverage)) - 1
    # The standard split-conformal order statistic includes an atom at
    # infinity.  Replacing it by the largest observed score would overstate
    # finite-sample validity when the audit set is too small for the requested
    # coverage level.
    if rank >= len(values):
        return float("inf")
    return float(values[max(rank, 0)])


@torch.no_grad()
def estimate_perturbation_floor(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    floor_fraction: float = 0.10,
) -> float:
    """Estimate the stabilized disagreement floor on proper training data."""

    if not hasattr(model, "raw_scale") or not hasattr(model, "set_scale_floor"):
        raise TypeError("model must expose raw_scale() and set_scale_floor()")
    if floor_fraction <= 0.0:
        raise ValueError("floor_fraction must be positive")
    disagreements: list[np.ndarray] = []
    model.eval()
    for state, action, viscosity, boundary, _ in loader:
        _, raw = model.raw_scale(
            state.to(device),
            action.to(device),
            viscosity.to(device),
            boundary.to(device),
        )
        # Exact boundary projections make boundary disagreement identically
        # zero. Exclude those known coordinates from the learned floor.
        interior = raw[:, 1:-1] if raw.shape[1] > 2 else raw
        disagreements.append(interior.cpu().numpy().reshape(-1))
    values = np.concatenate(disagreements)
    floor = max(1e-6, floor_fraction * float(np.median(values)))
    model.set_scale_floor(floor)
    return floor


@dataclass(frozen=True)
class OperatorCalibrator:
    multiplier: float
    coverage: float
    lipschitz: float
    norm_kind: str = "l2"

    def __post_init__(self) -> None:
        if self.norm_kind not in {"l2", "decision", "ellipsoid", "max"}:
            raise ValueError(f"Unknown norm: {self.norm_kind}")

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
        if self.norm_kind == "max":
            return value.abs().amax(dim=1)
        raise ValueError("Use standardized_score() for ellipsoidal set membership.")

    def standardized_score(
        self,
        error: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if self.norm_kind == "ellipsoid":
            return torch.sqrt(torch.mean((error / scale.clamp_min(1e-5)).square(), dim=1))
        if self.norm_kind == "max":
            # A max-type score yields simultaneous spatial coverage of every
            # coordinate of the one-step function-valued prediction.
            return (error.abs() / scale.clamp_min(1e-5)).amax(dim=1)
        if self.norm_kind != "ellipsoid":
            return self.functional_norm(error) / self.functional_norm(scale).clamp_min(1e-7)
        raise AssertionError("unreachable")

    def radius(
        self,
        scale: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a function-space radius or downstream support value."""

        if self.norm_kind in {"l2", "decision"}:
            return self.multiplier * self.functional_norm(scale)
        if reference is None:
            # Scalar radii are used only for diagnostics and the conservative
            # tube baseline. Decision-aware planning passes a sensitivity and
            # therefore uses the exact support function below.
            if self.norm_kind == "ellipsoid":
                return self.multiplier * torch.sqrt(torch.mean(scale.square(), dim=1))
            return self.multiplier * scale.amax(dim=1)
        if self.norm_kind == "ellipsoid":
            # Support of {delta: ||delta / scale||_{2,n} <= q} under <.,.>_n.
            return self.multiplier * torch.sqrt(torch.mean((reference * scale).square(), dim=1))
        # Support of the simultaneous box {|delta_j| <= q scale_j}.
        return self.multiplier * torch.mean(reference.abs() * scale, dim=1)

    def propagate(self, previous_radius: torch.Tensor, one_step_radius: torch.Tensor) -> torch.Tensor:
        return self.lipschitz * previous_radius + one_step_radius


def polyhedral_tightening_margin(
    calibrator: OperatorCalibrator,
    scale: torch.Tensor,
    constraint_matrix: torch.Tensor,
) -> torch.Tensor:
    r"""Return exact one-step margins for constraints ``A x <= b``.

    ``scale`` has shape ``(batch, state)`` and ``constraint_matrix`` has
    shape ``(constraints, state)``. For the simultaneous conformal box the
    margin is ``q |A| scale``. For the normalized ellipsoid it is
    ``q sqrt(n) ||A_i diag(scale)||_2``.
    """

    if scale.ndim != 2 or constraint_matrix.ndim != 2:
        raise ValueError("scale and constraint_matrix must be matrices")
    if scale.shape[1] != constraint_matrix.shape[1]:
        raise ValueError("state dimensions do not match")
    if calibrator.norm_kind == "max":
        return calibrator.multiplier * torch.einsum(
            "mn,bn->bm",
            constraint_matrix.abs(),
            scale,
        )
    if calibrator.norm_kind == "ellipsoid":
        state_dimension = scale.shape[1]
        weighted = constraint_matrix[None, :, :] * scale[:, None, :]
        return (
            calibrator.multiplier
            * np.sqrt(state_dimension)
            * torch.linalg.vector_norm(weighted, dim=2)
        )
    raise ValueError("exact coordinate-wise tightening requires max or ellipsoid geometry")


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
    errors, radii, membership_scores, decision_errors, decision_supports = [], [], [], [], []
    for state, action, viscosity, boundary, target in loader:
        state, action = state.to(device), action.to(device)
        viscosity, boundary = viscosity.to(device), boundary.to(device)
        target = target.to(device)
        mean, scale = model(state, action, viscosity, boundary)
        residual = target - mean
        l2_error = torch.sqrt(torch.mean(residual.square(), dim=1))
        errors.append(l2_error.cpu().numpy())
        membership_scores.append(calibrator.standardized_score(residual, scale).cpu().numpy())
        radii.append(calibrator.radius(scale).cpu().numpy())

        # One-step stage-cost sensitivity. This is a diagnostic of whether the
        # same calibrated set covers the linearized downstream cost error; it
        # is not used to recalibrate a different, decision-specific event.
        x = torch.linspace(0.0, 1.0, mean.shape[1], dtype=mean.dtype, device=mean.device)
        weight = 0.15 + 0.85 * torch.exp(-0.5 * ((x - 0.72) / 0.14).square())
        reference = 2.0 * weight[None, :] * mean
        decision_errors.append(
            torch.abs(torch.mean(reference * residual, dim=1)).cpu().numpy()
        )
        if calibrator.norm_kind in {"ellipsoid", "max"}:
            support = calibrator.radius(scale, reference)
        else:
            # Cauchy--Schwarz relaxation for scalar L2 tube baselines.
            support = calibrator.radius(scale) * torch.sqrt(
                torch.mean(reference.square(), dim=1)
            )
        decision_supports.append(support.cpu().numpy())
    error = np.concatenate(errors)
    radius = np.concatenate(radii)
    score = np.concatenate(membership_scores)
    decision_error = np.concatenate(decision_errors)
    decision_support = np.concatenate(decision_supports)
    coverage = float(np.mean(score <= calibrator.multiplier))
    return {
        "mean_functional_error": float(np.mean(error)),
        "coverage": coverage,
        "mean_radius": float(np.mean(radius)),
        "radius_error_correlation": float(np.corrcoef(radius, error)[0, 1]),
        "mean_standardized_score": float(np.mean(score)),
        "decision_linear_coverage": float(np.mean(decision_error <= decision_support)),
        "mean_decision_support": float(np.mean(decision_support)),
    }


@torch.no_grad()
def trajectory_scores(
    model: torch.nn.Module,
    trajectories,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return max-over-time-and-space rollout scores and mean local scales.

    The actions are those of the behavior-policy trajectories in the supplied
    dataset. Therefore these scores do not certify counterfactual MPC action
    sequences drawn from another distribution.
    """

    model.eval()
    all_scores: list[np.ndarray] = []
    all_scales: list[np.ndarray] = []
    total = int(trajectories.state.shape[0])
    horizon = int(trajectories.action.shape[1])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        truth = torch.as_tensor(
            trajectories.state[start:stop],
            dtype=torch.float32,
            device=device,
        )
        actions = torch.as_tensor(
            trajectories.action[start:stop],
            dtype=torch.float32,
            device=device,
        )
        viscosity = torch.as_tensor(
            trajectories.viscosity[start:stop],
            dtype=torch.float32,
            device=device,
        )
        boundary = torch.as_tensor(
            trajectories.boundary[start:stop],
            dtype=torch.float32,
            device=device,
        )
        predicted = truth[:, 0]
        sample_score = torch.zeros(stop - start, dtype=truth.dtype, device=device)
        scale_sum = torch.zeros_like(sample_score)
        for step in range(horizon):
            predicted, scale = model(
                predicted,
                actions[:, step],
                viscosity,
                boundary,
            )
            residual = truth[:, step + 1] - predicted
            step_score = (residual.abs() / scale.clamp_min(1e-5)).amax(dim=1)
            sample_score = torch.maximum(sample_score, step_score)
            scale_sum = scale_sum + scale.mean(dim=1)
        all_scores.append(sample_score.cpu().numpy())
        all_scales.append((scale_sum / horizon).cpu().numpy())
    return np.concatenate(all_scores), np.concatenate(all_scales)


def calibrate_trajectory_model(
    model: torch.nn.Module,
    trajectories,
    device: torch.device,
    coverage: float = 0.90,
    batch_size: int = 32,
) -> tuple[OperatorCalibrator, dict[str, float]]:
    """Calibrate a max-over-horizon-and-coordinate rollout band."""

    scores, scales = trajectory_scores(model, trajectories, device, batch_size)
    multiplier = _higher_quantile(scores, coverage)
    calibrator = OperatorCalibrator(multiplier, coverage, 1.0, "max")
    return calibrator, {
        "multiplier": multiplier,
        "coverage": coverage,
        "calibration_size": float(scores.size),
        "horizon": float(trajectories.action.shape[1]),
        "mean_uncalibrated_scale": float(np.mean(scales)),
        "event": "simultaneous_time_and_coordinate_rollout",
    }


def trajectory_coverage_metrics(
    model: torch.nn.Module,
    calibrator: OperatorCalibrator,
    trajectories,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, float]:
    scores, scales = trajectory_scores(model, trajectories, device, batch_size)
    return {
        "coverage": float(np.mean(scores <= calibrator.multiplier)),
        "mean_max_standardized_score": float(np.mean(scores)),
        "p90_max_standardized_score": float(np.quantile(scores, 0.90)),
        "mean_coordinate_radius": float(calibrator.multiplier * np.mean(scales)),
        "horizon": float(trajectories.action.shape[1]),
    }
