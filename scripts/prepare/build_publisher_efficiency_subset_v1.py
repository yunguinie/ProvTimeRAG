"""Freeze a label-independent stable-hash subset for efficiency profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_subset(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len({str(row["bundle_id"]) for row in rows}) != len(rows):
        raise ValueError("bundle_id values must be unique")
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(str(row["bundle_id"]).encode("utf-8")).hexdigest(),
            str(row["bundle_id"]),
        ),
    )
    return ranked[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = stable_subset(rows, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "frozen_before_efficiency_evaluation",
        "version": "publisher_efficiency_subset_v1",
        "selection": "smallest_sha256_bundle_id_label_independent",
        "input": str(args.input),
        "output": str(args.output),
        "input_bundles": len(rows),
        "selected_bundles": len(selected),
        "limit": args.limit,
        "sha256": {"input": sha256_file(args.input), "output": sha256_file(args.output)},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
