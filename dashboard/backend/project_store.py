from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import uuid
from .program import H2O_SOURCE, CIRCUIT_SOURCE


# The dashboard is served from the monorepo, so the source files shown in the
# editor must come from the same checkout that executes the task.  Keep this
# resolver local to the project store instead of accepting arbitrary paths from
# the browser.  Missing optional files are omitted rather than replaced by a
# generated placeholder.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _read_repository_files(paths: tuple[str, ...]) -> dict[str, str]:
    files: dict[str, str] = {}
    for relative_path in paths:
        path = (REPOSITORY_ROOT / relative_path).resolve()
        try:
            path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            # Defensive guard for future edits to the allow-list.
            continue
        if not path.is_file():
            continue
        try:
            files[relative_path.replace("\\", "/")] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Binary files and unreadable files do not belong in the source
            # editor.  The allow-list intentionally contains source only.
            continue
    return files


def _h2o_files() -> dict[str, str]:
    """Return the canonical H₂O entry plus the modules used by its run path."""

    files = {"main.py": H2O_SOURCE}
    files.update(
        _read_repository_files(
            (
                "applications/h2o-hybrid-aimd/scripts/run_aimd.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/workflows/run_aimd.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/simulation/aimd.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/core/factory.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/core/potential.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/quantum/qiskit_f2.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/execution/aimd_worker.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/execution/quantum_worker.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/execution/classical_actor.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/classical/torch_mlp.py",
                "applications/h2o-hybrid-aimd/single_h20_aimd/backends/force/central_finite_difference.py",
            )
        )
    )
    return files


def _circuit_files() -> dict[str, str]:
    """Return the circuit entry and the dashboard's real execution modules."""

    files = {"main.py": CIRCUIT_SOURCE}
    files.update(
        _read_repository_files(
            (
                "dashboard/backend/tasks/circuit.py",
                "dashboard/backend/circuit_parser.py",
                "dashboard/backend/circuit_runner.py",
            )
        )
    )
    return files

@dataclass
class Project:
    id: str
    name: str
    task_id: str
    files: dict[str,str] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    def as_dict(self, include_content=True):
        return {'id':self.id,'name':self.name,'task_id':self.task_id,'updated_at':self.updated_at,
                'files':[{'path':p,'content':c if include_content else None} for p,c in self.files.items()]}

def default_project():
    return Project('project-h2o-demo','H₂O AIMD','h2o-hybrid-aimd',_h2o_files())
PROJECTS = {'project-h2o-demo':default_project(),
            'project-circuit-demo':Project('project-circuit-demo','量子电路实验','quantum-circuit',_circuit_files())}
def create_project(name, task_id, files=None):
    project = Project('project-'+uuid.uuid4().hex[:10],name,task_id,files or {})
    PROJECTS[project.id] = project
    return project

def get_project(project_id):
    return PROJECTS.get(project_id)
