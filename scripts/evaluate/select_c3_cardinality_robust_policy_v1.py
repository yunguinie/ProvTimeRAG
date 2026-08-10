"""Select one C3 robust policy across seeds and two development domains."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DOMAINS = ("calibration", "external_dev")


def mean(rows: list[dict[str, Any]], policy: str, domain: str, metric: str) -> float:
    return statistics.mean(row["reports"][policy][domain][metric] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    seeds = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("input seeds must be unique")
    policy_names = set(rows[0]["reports"])
    if any(set(row["reports"]) != policy_names for row in rows[1:]):
        raise ValueError("policy grids differ across seeds")

    baseline = {
        domain: {
            metric: mean(rows, "independent", domain, metric)
            for metric in ("bundle_exact_match", "total_source_error")
        }
        for domain in DOMAINS
    }
    summaries = {}
    for policy in sorted(policy_names):
        domains = {}
        all_seed_deltas = []
        for domain in DOMAINS:
            bundle_values = [row["reports"][policy][domain]["bundle_exact_match"] for row in rows]
            source_values = [row["reports"][policy][domain]["total_source_error"] for row in rows]
            baseline_values = [row["reports"]["independent"][domain]["bundle_exact_match"] for row in rows]
            deltas = [value - base for value, base in zip(bundle_values, baseline_values, strict=True)]
            all_seed_deltas.extend(deltas)
            domains[domain] = {
                "bundle_exact_mean": statistics.mean(bundle_values),
                "bundle_exact_sample_std": statistics.stdev(bundle_values),
                "bundle_delta_mean": statistics.mean(deltas),
                "total_source_error_mean": statistics.mean(source_values),
                "correction_coverage_mean": mean(rows, policy, domain, "correction_coverage"),
                "improvements": sum(row["reports"][policy][domain]["improvements"] for row in rows),
                "regressions": sum(row["reports"][policy][domain]["regressions"] for row in rows),
            }
        mean_feasible = all(
            domains[domain]["bundle_exact_mean"] + 1e-12 >= baseline[domain]["bundle_exact_match"]
            and domains[domain]["total_source_error_mean"] <= baseline[domain]["total_source_error"] + 1e-12
            for domain in DOMAINS
        )
        strict_feasible = mean_feasible and min(all_seed_deltas) >= -1e-12
        summaries[policy] = {
            "configuration": rows[0]["policy_grid"][policy],
            "domains": domains,
            "mean_feasible": mean_feasible,
            "strict_feasible": strict_feasible,
            "minimum_seed_domain_delta": min(all_seed_deltas),
            "minimum_domain_mean_delta": min(domains[d]["bundle_delta_mean"] for d in DOMAINS),
            "mean_domain_delta": statistics.mean(domains[d]["bundle_delta_mean"] for d in DOMAINS),
        }

    strict = [name for name, item in summaries.items() if item["strict_feasible"]]
    pool = strict or [name for name, item in summaries.items() if item["mean_feasible"]]
    if not pool:
        raise RuntimeError("independent policy must always be feasible")
    selected = max(
        pool,
        key=lambda name: (
            summaries[name]["minimum_domain_mean_delta"],
            summaries[name]["mean_domain_delta"],
            -statistics.mean(
                summaries[name]["domains"][domain]["total_source_error_mean"]
                for domain in DOMAINS
            ),
            name != "independent",
        ),
    )
    result = {
        "status": "frozen_policy_candidate_before_new_external_test",
        "method": "cardinality_shift_robust_c3_cross_seed_selection_v1",
        "seeds": seeds,
        "development_domains": list(DOMAINS),
        "selected_policy": selected,
        "selected": summaries[selected],
        "baseline": baseline,
        "selection_pool": "strict_feasible" if strict else "mean_feasible",
        "policy_summaries": summaries,
        "policy": [
            "One policy is shared across all seeds.",
            "Strict feasibility requires no bundle-exact regression for any seed/domain and no mean source-error increase in either domain.",
            "Selection maximizes the worst domain mean delta before average delta.",
            "This policy must be frozen before evaluation on a new external test.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "selected_policy": selected,
        "selection_pool": result["selection_pool"],
        "selected": result["selected"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
