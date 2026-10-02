"""Executor contract shared by local and future Ray implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ray_quantum.framework.models import (
    InvocationHandle,
    InvocationResult,
    InvocationSpec,
)
from ray_quantum.framework.registry import ComponentRegistry


@runtime_checkable
class Executor(Protocol):
    """Submit, resolve, invoke, release, and close component calls."""

    @property
    def closed(self) -> bool:
        """Whether this executor permanently rejects new submissions."""

        ...

    @property
    def registry(self) -> ComponentRegistry:
        """Registry used for validation and component construction."""

        ...

    def submit(self, invocation: InvocationSpec) -> InvocationHandle:
        """Submit one invocation and return an executor-neutral handle."""

        ...

    def result(self, handle: InvocationHandle) -> InvocationResult:
        """Wait for and return one terminal invocation result."""

        ...

    def invoke(self, invocation: InvocationSpec) -> InvocationResult:
        """Submit and synchronously resolve one invocation."""

        ...

    def release(self, handle: InvocationHandle) -> None:
        """Release a completed result retained by this executor."""

        ...

    def close(self) -> None:
        """Stop accepting work and release executor-owned resources."""

        ...


__all__ = ["Executor"]
