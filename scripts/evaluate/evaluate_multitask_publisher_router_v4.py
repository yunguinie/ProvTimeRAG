"""Evaluate the frozen publisher router and label an external data contract.

This wrapper changes reporting metadata only.  All scoring is delegated to the
frozen v3 evaluator after removing the wrapper-only argument.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

from scripts.evaluate import evaluate_multitask_publisher_router_v3 as implementation


def pop_argument(argv: list[str], name: str) -> str:
    try:
        index = argv.index(name)
        value = argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{name} is required") from exc
    del argv[index : index + 2]
    return value


def argument(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{name} is required") from exc


def main() -> None:
    argv = sys.argv[1:]
    contract = pop_argument(argv, "--evaluation-contract")
    output = Path(argument(argv, "--output"))
    sys.argv = [sys.argv[0], *argv]
    with contextlib.redirect_stdout(io.StringIO()):
        implementation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    report["evaluation_contract"] = contract
    report["method"] = "multitask_publisher_identity_router_v3_frozen_external_holdout"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
