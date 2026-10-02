"""Extract verified Ubuntu runtime packages locally; never install/replace system libc."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.paths import data_path

ROOT = Path(os.environ.get('FUSION_QPERFSIM_RUNTIME', str(data_path('.qperfsim-runtime'))))
BASE = "https://archive.ubuntu.com/ubuntu/"
INDEX = BASE + "dists/noble-updates/main/binary-amd64/Packages.gz"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    contents = gzip.decompress(urllib.request.urlopen(INDEX, timeout=60).read()).decode()
    records = {}
    for paragraph in contents.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in paragraph.splitlines() if ": " in line and not line.startswith(" "))
        if fields.get("Package") in {"libc6", "libstdc++6", "libgcc-s1"} and fields.get("Architecture") == "amd64":
            records[fields["Package"]] = fields
    if len(records) != 3:
        raise RuntimeError("Runtime package index incomplete")
    manifest = []
    for name, entry in records.items():
        url = BASE + entry["Filename"]
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != entry["SHA256"]:
            raise RuntimeError(f"Package checksum mismatch: {name}")
        package = ROOT / Path(entry["Filename"]).name
        package.write_bytes(data)
        subprocess.run(["dpkg-deb", "-x", str(package), str(ROOT)], check=True)
        manifest.append({"package": name, "version": entry["Version"], "url": url, "sha256": entry["SHA256"]})
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
