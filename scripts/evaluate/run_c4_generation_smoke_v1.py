"""Run a resumable, schema-checked C4 generation smoke."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

RETRIABLE = {408, 409, 429, 500, 502, 503, 504}
STATES = {"SUPPORTED", "STALE", "WRONG_SOURCE", "CONTRADICTED", "PARTIALLY_SUPPORTED", "INSUFFICIENT"}


def endpoint(base: str) -> str:
    value = base.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions" if value.endswith("/v1") else f"{value}/v1/chat/completions"


def clean_error(value: Any) -> str:
    return " ".join(str(value).split())[:300]


def call_api(api_url: str, api_key: str, model: str, prompt: str, timeout: float) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a strict provenance-constrained answer generator. Return JSON only. Use only supplied evidence. Every factual statement must cite supplied evidence IDs. If evidence is stale, contradictory, insufficient, or cannot support the question, abstain or state the conflict."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 512,
        "enable_thinking": False,
    }
    request = urllib.request.Request(api_url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"], payload.get("usage", {})


def parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("response is not a JSON object")
    return value


def validate(value: dict[str, Any], candidate_count: int) -> dict[str, Any]:
    missing = {"answer", "citation_ids", "abstain", "state"} - set(value)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not isinstance(value["answer"], str):
        raise TypeError("answer must be a string")
    citations = value["citation_ids"]
    if not isinstance(citations, list) or not all(isinstance(item, str) and item.startswith("E") and item[1:].isdigit() for item in citations):
        raise TypeError("citation_ids must be a list of E-number strings")
    if any(int(item[1:]) < 1 or int(item[1:]) > candidate_count for item in citations):
        raise ValueError("citation ID outside supplied evidence")
    if not isinstance(value["abstain"], bool):
        raise TypeError("abstain must be boolean")
    if value["state"] not in STATES:
        raise ValueError(f"invalid state: {value['state']}")
    return value


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
        completed = {json.loads(line)["request_id"] for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()}
    processed = failures = 0
    for row in rows:
        request_id = row["request_id"]
        if request_id in completed:
            continue
        started = time.monotonic()
        error = ""
        for attempt in range(args.max_retries + 1):
            try:
                content, usage = call_api(endpoint(args.api_base), api_key, args.model, row["prompt"], args.timeout)
                prediction = validate(parse_json(content), len(row["candidates"]))
                result = {"request_id": request_id, "status": "complete", "model": args.model, "prediction": prediction, "gold": row["gold"], "usage": usage, "latency_seconds": round(time.monotonic() - started, 3)}
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                error = clean_error(exc)
                if attempt == args.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        if error:
            failures += 1
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"request_id": request_id, "status": "failed", "error": error}, ensure_ascii=False) + "\n")
            print(f"PROGRESS processed={processed}/{len(rows)} failures={failures}", flush=True)
    status = "complete" if failures == 0 else "complete_with_failures"
    print(json.dumps({"status": status, "records": len(rows), "processed": processed, "failures": failures, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
