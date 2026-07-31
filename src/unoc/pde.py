"""Controlled one-dimensional viscous Burgers equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BurgersConfig:
    grid_size: int = 64
    control_dt: float = 0.02
    solver_dt: float = 0.0002
    actuator_center: float = 0.68
    actuator_width: float = 0.12
    action_limit: float = 2.0


class BurgersSolver:
    r"""Explicit finite-difference solver.

    .. math::
       u_t + u u_x = \nu u_{xx} + a_t b(x),\qquad
       u(t,0)=b_L,\quad u(t,1)=b_R.
    """

    def __init__(self, config: BurgersConfig | None = None):
        self.config = config or BurgersConfig()
        self.x = np.linspace(0.0, 1.0, self.config.grid_size)
        profile = np.exp(
            -0.5 * ((self.x - self.config.actuator_center) / self.config.actuator_width) ** 2
        )
        profile[0] = profile[-1] = 0.0
        self.actuator = profile / np.max(profile)

    def random_state(
        self,
        rng: np.random.Generator,
        left: float,
        right: float,
        amplitude: tuple[float, float] = (0.15, 0.55),
    ) -> np.ndarray:
        """Draw a smooth state satisfying non-homogeneous Dirichlet conditions."""

        state = left * (1.0 - self.x) + right * self.x
        for mode in range(1, 5):
            coefficient = rng.uniform(*amplitude) * rng.normal() / mode
            state += coefficient * np.sin(mode * np.pi * self.x)
        state[0], state[-1] = left, right
        return state.astype(np.float64)

    def step(
        self,
        state: np.ndarray,
        action: float,
        viscosity: float,
        left: float,
        right: float,
        actuator_gain: float = 1.0,
    ) -> np.ndarray:
        """Advance one control interval with upwind convection."""

        cfg = self.config
        action = float(np.clip(action, -cfg.action_limit, cfg.action_limit))
        substeps = int(np.ceil(cfg.control_dt / cfg.solver_dt))
        dt = cfg.control_dt / substeps
        dx = 1.0 / (cfg.grid_size - 1)
        if dt * viscosity / (dx * dx) > 0.49:
            raise ValueError("Diffusion CFL condition violated.")

        u = np.asarray(state, dtype=np.float64).copy()
        for _ in range(substeps):
            backward = (u[1:-1] - u[:-2]) / dx
            forward = (u[2:] - u[1:-1]) / dx
            upwind = np.where(u[1:-1] >= 0.0, backward, forward)
            laplacian = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
            interior = (
                u[1:-1]
                + dt
                * (
                    -u[1:-1] * upwind
                    + viscosity * laplacian
                    + actuator_gain * action * self.actuator[1:-1]
                )
            )
            u[1:-1] = interior
            u[0], u[-1] = left, right
        return u

    @staticmethod
    def stage_cost(state: np.ndarray, action: float, control_weight: float = 0.002) -> float:
        x = np.linspace(0.0, 1.0, state.size)
        weight = 0.15 + 0.85 * np.exp(-0.5 * ((x - 0.72) / 0.14) ** 2)
        return float(np.mean(weight * np.square(state)) + control_weight * action * action)
