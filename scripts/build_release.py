"""Create a small source archive; never include local datasets, weights or environments."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from check_repository import ROOT, check, release_files


def build(output):
    check()
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = release_files()
    manifest = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo("Neural-BVI/" + path.relative_to(ROOT).as_posix(),
                                   date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("Neural-BVI/RELEASE_MANIFEST.json", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(manifest, indent=2) + "\n")
    return {"archive": str(output.resolve()), "files": len(files), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/neural-bvi-source.zip")
    print(json.dumps(build(parser.parse_args().output), indent=2))
