"""Paired relative HDR evaluation without neural models or GPU access."""

from .metrics import aggregate_cases, evaluate_case
from .runner import run_evaluation

__all__ = ["evaluate_case", "aggregate_cases", "run_evaluation"]
