"""PivotQ: Python CPU tasks and explicit quantum execution backends."""

from typing import TYPE_CHECKING, Any

from .errors import PivotQError
from .refs import ResultRef
from .runtime import Runtime
from .quantum import QuantumBackend, QuantumResult
from .components import ComponentHandle, ComponentSpec
from .workflow import NodeRef, Workflow, WorkflowRun, WorkflowSubmissionError
from .observability import ExecutionReport, InvocationStatus
from .providers import BackendCapabilities, QuantumProvider, QuantumRequest, ProviderResult

if TYPE_CHECKING:
    from .circuit import Parameter as Parameter, QuantumCircuit as QuantumCircuit

__version__ = "0.1.0.dev0"

__all__ = [
    "Runtime", "ResultRef", "QuantumBackend", "QuantumResult", "PivotQError",
    "ComponentSpec", "ComponentHandle", "Workflow", "WorkflowRun", "NodeRef",
    "WorkflowSubmissionError", "ExecutionReport", "InvocationStatus",
    "BackendCapabilities", "QuantumProvider", "QuantumRequest", "ProviderResult", "__version__",
    "QuantumCircuit", "Parameter",
]


def __getattr__(name: str) -> Any:
    """Load circuit types only when explicitly requested by the caller."""
    if name in {"QuantumCircuit", "Parameter"}:
        from . import circuit

        value = getattr(circuit, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
