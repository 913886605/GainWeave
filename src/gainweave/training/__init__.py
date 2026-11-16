"""User-data preparation and the frozen-foundation, smooth-head training recipe."""

from .data import prepare_dataset


def run_training(config_path, data_root, output_dir=None, device="cuda", updates=None):
    """Run explicit training; importing this package never loads a neural model."""
    from .engine import run_training as run

    return run(config_path, data_root, output_dir=output_dir, device=device, updates=updates)


__all__ = ["prepare_dataset", "run_training"]
