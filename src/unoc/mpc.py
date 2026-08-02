"""Nominal and uncertainty-tube robust model-predictive control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .calibration import OperatorCalibrator


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 8
    candidates: int = 128
    elites: int = 16
    iterations: int = 4
    action_limit: float = 2.0
    control_weight: float = 0.002


def _decision_norm(value: torch.Tensor) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, value.shape[1], dtype=value.dtype, device=value.device)
    weight = 0.15 + 0.85 * torch.exp(-0.5 * ((x - 0.72) / 0.14).square())
    return torch.sqrt(torch.mean(weight[None, :] * value.square(), dim=1))


def _sequence_cost(
    model: torch.nn.Module,
    calibrator: OperatorCalibrator,
    initial_state: torch.Tensor,
    sequences: torch.Tensor,
    viscosity: float,
    boundary: tuple[float, float],
    robust: bool,
    control_weight: float,
) -> torch.Tensor:
    if robust and calibrator.norm_kind in {"ellipsoid", "max"}:
        return _adjoint_robust_sequence_cost(
            model,
            calibrator,
            initial_state,
            sequences,
            viscosity,
            boundary,
            control_weight,
        )
    with torch.no_grad():
        return _tube_sequence_cost(
            model,
            calibrator,
            initial_state,
            sequences,
            viscosity,
            boundary,
            robust,
            control_weight,
        )


def _tube_sequence_cost(
    model: torch.nn.Module,
    calibrator: OperatorCalibrator,
    initial_state: torch.Tensor,
    sequences: torch.Tensor,
    viscosity: float,
    boundary: tuple[float, float],
    robust: bool,
    control_weight: float,
) -> torch.Tensor:
    count, horizon = sequences.shape
    state = initial_state[None, :].expand(count, -1)
    nu = torch.full((count,), viscosity, dtype=state.dtype, device=state.device)
    bc = torch.tensor(boundary, dtype=state.dtype, device=state.device)[None, :].expand(count, -1)
    tube = torch.zeros(count, dtype=state.dtype, device=state.device)
    total = torch.zeros_like(tube)
    for step in range(horizon):
        action = sequences[:, step]
        mean, scale = model(state, action, nu, bc)
        state_norm = _decision_norm(mean)
        if robust:
            tube = calibrator.propagate(tube, calibrator.radius(scale))
        total += (state_norm + tube).square()
        total += control_weight * action.square()
        state = mean
    return total


def _adjoint_robust_sequence_cost(
    model: torch.nn.Module,
    calibrator: OperatorCalibrator,
    initial_state: torch.Tensor,
    sequences: torch.Tensor,
    viscosity: float,
    boundary: tuple[float, float],
    control_weight: float,
) -> torch.Tensor:
    """First-order robust counterpart using full finite-horizon sensitivities.

    The calibrator supplies the exact support of either the normalized
    ellipsoid or the simultaneous coordinate box in the adjoint direction.
    """

    count, horizon = sequences.shape
    state = initial_state[None, :].expand(count, -1)
    nu = torch.full((count,), viscosity, dtype=state.dtype, device=state.device)
    bc = torch.tensor(boundary, dtype=state.dtype, device=state.device)[None, :].expand(count, -1)
    predicted_states: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    nominal = torch.zeros(count, dtype=state.dtype, device=state.device)
    with torch.enable_grad():
        for step in range(horizon):
            action = sequences[:, step]
            state, scale = model(state, action, nu, bc)
            predicted_states.append(state)
            scales.append(scale)
            nominal = nominal + _decision_norm(state).square()
            nominal = nominal + control_weight * action.square()
        adjoints = torch.autograd.grad(
            nominal.sum(),
            predicted_states,
            retain_graph=False,
            create_graph=False,
        )
        robust_support = torch.zeros_like(nominal)
        for scale, adjoint in zip(scales, adjoints):
            # autograd returns the Euclidean coordinate gradient dJ/dx_j.
            # The paper uses <lambda, delta>_n = n^{-1} sum_j lambda_j delta_j,
            # hence lambda_j = n * dJ/dx_j.
            normalized_inner_product_adjoint = adjoint.detach() * scale.shape[1]
            robust_support = robust_support + calibrator.radius(
                scale.detach(),
                normalized_inner_product_adjoint,
            )
    return (nominal.detach() + robust_support).detach()


def cem_action(
    model: torch.nn.Module,
    calibrator: OperatorCalibrator,
    state: np.ndarray,
    viscosity: float,
    boundary: tuple[float, float],
    config: CEMConfig,
    device: torch.device,
    robust: bool,
    seed: int,
) -> float:
    """Return the first action from a CEM-optimized open-loop sequence."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    mean = torch.zeros(config.horizon, device=device)
    std = torch.full((config.horizon,), 0.65, device=device)
    initial = torch.as_tensor(state, dtype=torch.float32, device=device)
    for _ in range(config.iterations):
        noise = torch.randn(
            config.candidates,
            config.horizon,
            generator=generator,
            device=device,
        )
        sequences = torch.clamp(
            mean[None, :] + std[None, :] * noise,
            -config.action_limit,
            config.action_limit,
        )
        costs = _sequence_cost(
            model,
            calibrator,
            initial,
            sequences,
            viscosity,
            boundary,
            robust,
            config.control_weight,
        )
        elite = sequences[torch.topk(costs, config.elites, largest=False).indices]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(0.05)
    return float(mean[0].cpu())
