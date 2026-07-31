"""Training utilities for probabilistic one-step operator world models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    scale_floor: float = 1e-4
    uncertainty_quantile: float = 0.90


def probabilistic_loss(
    mean: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    scale_floor: float,
    uncertainty_quantile: float = 0.90,
) -> torch.Tensor:
    scale = scale.clamp_min(scale_floor)
    error = target - mean
    relative = torch.mean(error.square()) / torch.mean(target.square()).clamp_min(1e-6)
    # Train the uncertainty head as a conditional residual quantile. Detaching
    # the residual prevents the mean from worsening merely to ease scale fitting.
    quantile_residual = error.detach().abs() - scale
    pinball = torch.maximum(
        uncertainty_quantile * quantile_residual,
        (uncertainty_quantile - 1.0) * quantile_residual,
    ).mean()
    normalized_pinball = pinball / target.detach().abs().mean().clamp_min(1e-3)
    return relative + 0.50 * normalized_pinball


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    for epoch in range(config.epochs):
        model.train()
        train_total = 0.0
        for state, action, viscosity, boundary, target in train_loader:
            state, action = state.to(device), action.to(device)
            viscosity, boundary = viscosity.to(device), boundary.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, scale = model(state, action, viscosity, boundary)
            loss = probabilistic_loss(
                mean,
                scale,
                target,
                config.scale_floor,
                config.uncertainty_quantile,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * state.shape[0]

        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for state, action, viscosity, boundary, target in validation_loader:
                state, action = state.to(device), action.to(device)
                viscosity, boundary = viscosity.to(device), boundary.to(device)
                target = target.to(device)
                mean, scale = model(state, action, viscosity, boundary)
                loss = probabilistic_loss(
                    mean,
                    scale,
                    target,
                    config.scale_floor,
                    config.uncertainty_quantile,
                )
                validation_total += float(loss) * state.shape[0]
        train_loss = train_total / len(train_loader.dataset)
        validation_loss = validation_total / len(validation_loader.dataset)
        history.append(
            {"epoch": float(epoch + 1), "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
