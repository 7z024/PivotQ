"""Inspect the first and last frames of the imported H2O trajectory."""

import csv
import math
import sys
from pathlib import Path


data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("imported")


def rows(filename):
    with (data_dir / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atom(position, name):
    return [float(position[f"{name}_{axis}_A"]) for axis in "xyz"]


logs = rows("md_log.csv")
positions = rows("positions.csv")
if not logs or len(logs) != len(positions):
    raise ValueError("Energy log and position file must have the same nonzero frame count")

for index in (0, -1):
    log, position = logs[index], positions[index]
    if log["step"] != position["step"]:
        raise ValueError("Energy log and position steps do not match")
    oxygen, hydrogen_1, hydrogen_2 = (atom(position, name) for name in ("O", "H1", "H2"))
    bond_1 = [h - o for h, o in zip(hydrogen_1, oxygen)]
    bond_2 = [h - o for h, o in zip(hydrogen_2, oxygen)]
    length_1, length_2 = math.dist(oxygen, hydrogen_1), math.dist(oxygen, hydrogen_2)
    cosine = sum(a * b for a, b in zip(bond_1, bond_2)) / (length_1 * length_2)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    print(f"step={log['step']} time_fs={float(log['time_fs']):.1f}")
    print(f"  oh1_A={length_1:.4f} oh2_A={length_2:.4f} hoh_deg={angle:.2f}")
    print(f"  total_eV={float(log['total_energy_eV']):.4f}")

change = float(logs[-1]["total_energy_eV"]) - float(logs[0]["total_energy_eV"])
print(f"delta_total_eV={change:.4f}")
