"""Standalone H₂O hybrid-potential and AIMD package."""

from .core.factory import build_hybrid_potential, load_hybrid_potential

__all__ = ["build_hybrid_potential", "load_hybrid_potential"]
