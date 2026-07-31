"""Decision-calibrated dynamics ambiguity sets for robust PDE control."""

from .calibration import OperatorCalibrator, calibrate_model
from .models import OperatorWorldModel, build_model
from .mpc import CEMConfig, cem_action
from .pde import BurgersConfig, BurgersSolver

__all__ = [
    "BurgersConfig",
    "BurgersSolver",
    "CEMConfig",
    "OperatorCalibrator",
    "OperatorWorldModel",
    "build_model",
    "calibrate_model",
    "cem_action",
]

__version__ = "0.1.0"
