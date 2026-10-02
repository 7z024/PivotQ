from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.integration.fusion_framework.submit_job import main


if __name__ == "__main__":
    raise SystemExit(main())
