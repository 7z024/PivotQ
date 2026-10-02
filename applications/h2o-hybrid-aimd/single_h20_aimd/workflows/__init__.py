"""Executable workflows for the standalone H₂O project."""

from .evaluate import run_evaluation
from .run_aimd import run_aimd

__all__ = ["run_aimd", "run_evaluation"]
