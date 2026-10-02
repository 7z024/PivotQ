"""Run from the copied package: python run_qcontrol.py plan|train|aimd|all|plots."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "framework/src"), str(ROOT / "aimd")]
os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

from single_h20_aimd.workflows.qcontrol_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
