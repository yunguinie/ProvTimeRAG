"""Create leakage-controlled publisher-identity Source-Swap splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from provtimerag.data import read_bundles_jsonl, validate_dataset, write_bundles_jsonl


def bundle_documents(bundle: Any) -> set[str]:
    return {
        candidate.evidence.document_id
        for group in bundle.groups
        for candidate in group.candidates
    }


def dataset_documents(bundles: list[Any]) -> set[str]:
    return {document for bundle in bundles for document in bundle_documents(bundle)}


def dataset_queries(bundles: list[Any]) -> set[str]:
    return {bundle.query.text.strip().casefold() for bundle in bundles}


def filter_document_overlap(
    bundles: list[Any], protected_documents: set[str]
) -> tuple[list[Any], list[dict[str, Any]]]:
    kept = []
    removed = []
    for bundle in bundles:
        overlap = sorted(bundle_documents(bundle) & protected_documents)
        if overlap:
            removed.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "query_id": bundle.query.query_id,
                    "overlapping_documents": overlap,
                }
            )
        else:
            kept.append(bundle)
    return kept, removed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap_report(train: list[Any], calibration: list[Any], dev: list[Any]) -> dict[str, int]:
    train_docs = dataset_documents(train)
    calibration_docs = dataset_documents(calibration)
    dev_docs = dataset_documents(dev)
    train_queries = dataset_queries(train)
    calibration_queries = dataset_queries(calibration)
    dev_queries = dataset_queries(dev)
    return {
        "train_calibration_documents": len(train_docs & calibration_docs),
        "train_dev_documents": len(train_docs & dev_docs),
        "calibration_dev_documents": len(calibration_docs & dev_docs),
        "train_calibration_queries": len(train_queries & calibration_queries),
        "train_dev_queries": len(train_queries & dev_queries),
        "calibration_dev_queries": len(calibration_queries & dev_queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--calibration-input", type=Path, required=True)
    parser.add_argument("--dev-input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    train = list(read_bundles_jsonl(args.train_input))
    calibration_raw = list(read_bundles_jsonl(args.calibration_input))
    dev_raw = list(read_bundles_jsonl(args.dev_input))
    pre_overlap = overlap_report(train, calibration_raw, dev_raw)

    train_docs = dataset_documents(train)
    calibration, removed_calibration = filter_document_overlap(
        calibration_raw, train_docs
    )
    protected_dev_docs = train_docs | dataset_documents(calibration)
    dev, removed_dev = filter_document_overlap(dev_raw, protected_dev_docs)
    post_overlap = overlap_report(train, calibration, dev)
    if any(post_overlap.values()):
        raise RuntimeError(f"post-cleaning leakage remains: {post_overlap}")
    for rows in (train, calibration, dev):
        validate_dataset(rows)

    outputs = {
        "train": (args.train_output, train),
        "calibration": (args.calibration_output, calibration),
        "dev": (args.dev_output, dev),
    }
    for path, rows in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_bundles_jsonl(path, rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "complete",
        "version": "source_swap_publisher_identity_v3_leakage_controlled",
        "priority_policy": "retain train; filter calibration against train; filter dev against train and cleaned calibration",
        "inputs": {
            "train": str(args.train_input),
            "calibration": str(args.calibration_input),
            "dev": str(args.dev_input),
        },
        "outputs": {name: str(path) for name, (path, _) in outputs.items()},
        "counts": {
            "train": len(train),
            "calibration_before": len(calibration_raw),
            "calibration_after": len(calibration),
            "calibration_removed": len(removed_calibration),
            "dev_before": len(dev_raw),
            "dev_after": len(dev),
            "dev_removed": len(removed_dev),
        },
        "pre_cleaning_overlap": pre_overlap,
        "post_cleaning_overlap": post_overlap,
        "removed": {
            "calibration": removed_calibration,
            "dev": removed_dev,
        },
        "sha256": {
            name: sha256_file(path) for name, (path, _) in outputs.items()
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
