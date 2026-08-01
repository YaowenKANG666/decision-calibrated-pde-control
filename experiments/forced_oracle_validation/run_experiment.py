"""Run the persistent-forcing task-validity experiment from a source checkout."""

from pathlib import Path

from unoc.forced_control import run_task_validation


if __name__ == "__main__":
    run_task_validation(Path("results/forced_oracle_validation"))
