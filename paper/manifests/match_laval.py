"""Index the identical Laval panorama files in a locally licensed, unpacked copy."""
import argparse
import csv
import hashlib
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_manifest(name):
    with (HERE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def required_sources(scope):
    q80 = {r["source_id"] for r in read_manifest("laval_dev_cases.csv") if r["q80_id"]}
    required = {}
    for r in read_manifest("laval_views.csv"):
        if scope == "dev" and r["split"] != "dev":
            continue
        if scope == "q80" and r["source_id"] not in q80:
            continue
        required[r["source_sha256"]] = (r["source_id"], int(r["source_bytes"]))
    return required


def match(root, required):
    allowed_sizes = {size for _, size in required.values()}
    matches = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() != ".exr" or not path.is_file() or path.stat().st_size not in allowed_sizes:
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in required:
            source_id, _ = required[digest]
            matches.setdefault(source_id, path.relative_to(root).as_posix())
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory of original EXR panoramas")
    parser.add_argument("--scope", choices=("q80", "dev", "all"), default="dev")
    parser.add_argument("--output", type=Path, required=True, help="Local CSV with paths relative to --root")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("The source root must be an existing directory.")
    required = required_sources(args.scope)
    found = match(args.root, required)
    missing = sorted({sid for sid, _ in required.values()} - found.keys())
    if missing:
        parser.exit(1, f"{len(missing)} of {len(required)} source files do not match. "
                       f"First unmatched IDs: {', '.join(missing[:5])}.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False,
                                     newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["source_id", "local_path"])
        writer.writerows(sorted(found.items()))
    os.replace(stream.name, args.output)
    print(f"Indexed {len(found)} matching source files for {args.scope} in {args.output}.")


if __name__ == "__main__":
    main()
