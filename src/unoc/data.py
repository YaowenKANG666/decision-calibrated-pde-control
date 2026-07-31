"""Dataset generation for in-distribution and shifted Burgers regimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .pde import BurgersSolver


@dataclass(frozen=True)
class Regime:
    viscosity: tuple[float, float]
    boundary: tuple[float, float]
    amplitude: tuple[float, float] = (0.15, 0.55)
    actuator_gain: tuple[float, float] = (0.60, 1.40)


TRAIN_REGIME = Regime((0.015, 0.025), (-0.03, 0.03))
PARAMETER_SHIFT = Regime((0.006, 0.012), (-0.03, 0.03))
BOUNDARY_SHIFT = Regime((0.015, 0.025), (-0.16, 0.16))
COMBINED_SHIFT = Regime(
    (0.006, 0.012),
    (-0.16, 0.16),
    (0.30, 0.75),
    (-0.50, 1.50),
)


@dataclass(frozen=True)
class TransitionArrays:
    state: np.ndarray
    action: np.ndarray
    viscosity: np.ndarray
    boundary: np.ndarray
    next_state: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            state=self.state,
            action=self.action,
            viscosity=self.viscosity,
            boundary=self.boundary,
            next_state=self.next_state,
        )

    @classmethod
    def load(cls, path: Path) -> "TransitionArrays":
        with np.load(path) as data:
            return cls(*(data[key] for key in ("state", "action", "viscosity", "boundary", "next_state")))


def generate_transitions(
    solver: BurgersSolver,
    size: int,
    regime: Regime,
    seed: int,
) -> TransitionArrays:
    rng = np.random.default_rng(seed)
    n = solver.config.grid_size
    states = np.empty((size, n), dtype=np.float32)
    next_states = np.empty_like(states)
    actions = rng.uniform(-solver.config.action_limit, solver.config.action_limit, size).astype(
        np.float32
    )
    viscosities = rng.uniform(*regime.viscosity, size).astype(np.float32)
    boundaries = rng.uniform(*regime.boundary, (size, 2)).astype(np.float32)
    actuator_gains = rng.uniform(*regime.actuator_gain, size)
    for index in range(size):
        state = solver.random_state(
            rng,
            float(boundaries[index, 0]),
            float(boundaries[index, 1]),
            regime.amplitude,
        )
        states[index] = state
        next_states[index] = solver.step(
            state,
            float(actions[index]),
            float(viscosities[index]),
            float(boundaries[index, 0]),
            float(boundaries[index, 1]),
            float(actuator_gains[index]),
        )
    return TransitionArrays(states, actions, viscosities, boundaries, next_states)


class TransitionDataset(Dataset):
    def __init__(self, arrays: TransitionArrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return int(self.arrays.state.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(self.arrays.state[index]),
            torch.tensor(self.arrays.action[index]),
            torch.tensor(self.arrays.viscosity[index]),
            torch.from_numpy(self.arrays.boundary[index]),
            torch.from_numpy(self.arrays.next_state[index]),
        )
