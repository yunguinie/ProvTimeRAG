"""Resumable C4 smoke runner; failed records remain eligible for retry."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
import urllib.error
from pathlib import Path

from scripts.evaluate.run_c4_generation_smoke_v1 import (
    RETRIABLE,
    call_api,
    clean_error,
    endpoint,
    parse_json,
    validate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("C4_GENERATOR_MODEL", os.getenv("SOURCE_STATE_MODEL", "")))
    parser.add_argument("--api-base", default=os.getenv("SOURCE_STATE_API_BASE", ""))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    api_key = os.getenv("SOURCE_STATE_API_KEY", "")
    if not args.model or not args.api_base or not api_key:
        raise ValueError("model, SOURCE_STATE_API_BASE, and SOURCE_STATE_API_KEY are required")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            item["request_id"]
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for item in [json.loads(line)]
            if item.get("status") == "complete"
        }
    processed = failures = 0
    for row in rows:
        if row["request_id"] in completed:
            continue
        started = time.monotonic()
        error = ""
        for attempt in range(args.max_retries + 1):
            try:
                content, usage = call_api(endpoint(args.api_base), api_key, args.model, row["prompt"], args.timeout)
                prediction = validate(parse_json(content), len(row["candidates"]))
                result = {"request_id": row["request_id"], "status": "complete", "model": args.model, "prediction": prediction, "gold": row["gold"], "usage": usage, "latency_seconds": round(time.monotonic() - started, 3)}
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1
                print(f"PROGRESS processed={processed}/{len(rows)} failures={failures}", flush=True)
                error = ""
                break
            except urllib.error.HTTPError as exc:
                error = f"HTTP {exc.code}"
                if exc.code not in RETRIABLE or attempt == args.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
            except (
                http.client.RemoteDisconnected,
                ConnectionError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                error = clean_error(exc)
                if attempt == args.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        if error:
            failures += 1
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"request_id": row["request_id"], "status": "failed", "error": error}, ensure_ascii=False) + "\n")
            print(f"PROGRESS processed={processed}/{len(rows)} failures={failures}", flush=True)
    status = "complete" if failures == 0 else "complete_with_failures"
    print(json.dumps({"status": status, "records": len(rows), "processed": processed, "failures": failures, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
