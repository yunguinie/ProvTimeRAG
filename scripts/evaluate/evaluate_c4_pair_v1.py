"""Evaluate paired C4 raw-evidence and C2-routed generation outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {row["request_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def tokens(value: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (value or "").lower())


def f1(predicted: str | None, gold: str | None) -> float:
    left, right = Counter(tokens(predicted)), Counter(tokens(gold))
    overlap = sum((left & right).values())
    return 2 * overlap / (sum(left.values()) + sum(right.values())) if left and right else 0.0


def evaluate(input_path: Path, output_path: Path) -> dict[str, Any]:
    inputs = load(input_path)
    outputs = [row for row in (json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip())]
    values: list[dict[str, Any]] = []
    for output in outputs:
        base_id = output["request_id"].removesuffix("-routed")
        row = inputs.get(output["request_id"]) or inputs[base_id]
        prediction = output["prediction"]
        candidates = row["candidates"]
        gold = set(row["gold"]["gold_evidence_ids"])
        cited: set[str] = set()
        for label in prediction.get("citation_ids", []):
            match = re.fullmatch(r"E(\d+)", str(label))
            if match and 1 <= int(match.group(1)) <= len(candidates):
                cited.add(candidates[int(match.group(1)) - 1]["evidence_id"])
        values.append(
            {
                "answer_f1": f1(prediction.get("answer"), row["gold"]["answer"]),
                "citation_precision": len(cited & gold) / len(cited) if cited else 0.0,
                "citation_recall": len(cited & gold) / len(gold) if gold else 0.0,
                "citation_hit": bool(cited & gold),
                "abstain": bool(prediction.get("abstain")),
                "latency": float(output.get("latency_seconds", 0.0)),
                "tokens": output.get("usage", {}).get("total_tokens", 0),
                "state": prediction.get("state"),
            }
        )
    count = len(values)
    return {
        "records": count,
        "answer_token_f1": sum(value["answer_f1"] for value in values) / count,
        "citation_precision": sum(value["citation_precision"] for value in values) / count,
        "citation_recall": sum(value["citation_recall"] for value in values) / count,
        "citation_hit_rate": sum(value["citation_hit"] for value in values) / count,
        "abstention_rate": sum(value["abstain"] for value in values) / count,
        "mean_latency_seconds": sum(value["latency"] for value in values) / count,
        "mean_total_tokens": sum(value["tokens"] for value in values) / count,
        "states": dict(Counter(value["state"] for value in values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--routed-input", type=Path, required=True)
    parser.add_argument("--routed-output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "complete", "raw": evaluate(args.raw_input, args.raw_output), "routed": evaluate(args.routed_input, args.routed_output)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
