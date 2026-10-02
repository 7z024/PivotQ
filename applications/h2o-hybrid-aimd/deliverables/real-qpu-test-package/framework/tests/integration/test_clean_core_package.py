"""P8.1 offline active-wheel content and cold-import gate."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import venv
import zipfile

from setuptools.build_meta import build_wheel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CleanCorePackageTest(unittest.TestCase):
    def test_offline_wheel_install_and_cold_import_are_application_neutral(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ray-quantum-p81-baseline-") as root_text:
            root = Path(root_text)
            project = root / "project"
            source = project / "src"
            distribution = root / "dist"
            project.mkdir()
            source.mkdir()
            distribution.mkdir()
            shutil.copy2(_REPOSITORY_ROOT / "pyproject.toml", project)
            shutil.copy2(_REPOSITORY_ROOT / "README.md", project)
            shutil.copytree(
                _REPOSITORY_ROOT / "src" / "ray_quantum",
                source / "ray_quantum",
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", "*.pyo", "*.zip"
                ),
            )
            historical_source = _REPOSITORY_ROOT / "src" / "ray_quantum_ir"
            if historical_source.is_dir():
                shutil.copytree(
                    historical_source,
                    source / "ray_quantum_ir",
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", "*.pyo", "*.zip"
                    ),
                )

            previous = Path.cwd()
            try:
                os.chdir(project)
                wheel_name = build_wheel(str(distribution))
            finally:
                os.chdir(previous)
            wheel_path = distribution / wheel_name
            self.assertTrue(wheel_path.is_file())

            with zipfile.ZipFile(wheel_path) as archive:
                names = tuple(archive.namelist())
                self.assertIn("ray_quantum/__init__.py", names)
                self.assertIn("ray_quantum/jobs/driver.py", names)
                self.assertIn("ray_quantum/observability/events.py", names)
                self.assertIn("ray_quantum/qpu_integration/service.py", names)
                self.assertIn(
                    "ray_quantum/qpu_integration/qiskit_to_qcis.py",
                    names,
                )
                self.assertFalse(
                    any(name.startswith("ray_quantum_ir/") for name in names)
                )
                self.assertFalse(
                    any(
                        name.startswith("ray_quantum/ir/")
                        or name.startswith("ray_quantum/source/")
                        for name in names
                    )
                )
                self.assertFalse(
                    any(
                        "__pycache__" in name
                        or name.endswith((".pyc", ".pyo", ".zip"))
                        for name in names
                    )
                )
                metadata_name = next(
                    name for name in names if name.endswith(".dist-info/METADATA")
                )
                metadata = archive.read(metadata_name).decode("utf-8")
                lowered = metadata.lower()
                self.assertNotIn("requires-dist: ase", lowered)
                self.assertNotIn("requires-dist: aimd", lowered)
                self.assertNotIn("requires-dist: m" + "lir", lowered)
                self.assertNotIn("requires-dist: l" + "lvm", lowered)
                self.assertNotIn("provides-extra: openqasm3", lowered)
                self.assertNotIn("qiskit-qasm3-import", lowered)

            environment = root / "venv"
            venv.EnvBuilder(with_pip=False).create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            located = subprocess.run(
                (
                    str(python),
                    "-c",
                    "import sysconfig; print(sysconfig.get_paths()['purelib'])",
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(located.returncode, 0, located.stderr)
            installed = Path(located.stdout.strip())
            with zipfile.ZipFile(wheel_path) as archive:
                archive.extractall(installed)

            script = """
import importlib.util
from pathlib import Path
import sys

import ray_quantum
import ray_quantum.jobs
import ray_quantum.observability
import ray_quantum.qpu_integration
import ray_quantum.qpu_integration.qiskit_to_qcis

installed = Path(ray_quantum.__file__).resolve()
assert "site-packages" in installed.parts, installed
assert importlib.util.find_spec("ray_quantum.ir") is None
assert importlib.util.find_spec("ray_quantum.source") is None
assert importlib.util.find_spec("ray_quantum_ir") is None
assert not any(name == "ray" or name.startswith("ray.") for name in sys.modules)
assert not any(name == "qiskit" or name.startswith("qiskit.") for name in sys.modules)
assert not any(name == "pyqos" or name.startswith("pyqos.") for name in sys.modules)
assert "ase" not in sys.modules
assert not any(name == "aimd" or name.startswith("aimd.") for name in sys.modules)
print(installed)
"""
            clean_environment = dict(os.environ)
            clean_environment.pop("PYTHONPATH", None)
            clean_environment["PYTHONNOUSERSITE"] = "1"
            imported = subprocess.run(
                (str(python), "-c", script),
                cwd=root,
                env=clean_environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            self.assertTrue(imported.stdout.strip().startswith(str(installed)))


if __name__ == "__main__":
    unittest.main()
