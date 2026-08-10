"""Final metric-corrected frozen C3 evaluation entry point."""

from __future__ import annotations

from scripts.evaluate.c3_visible_tie_breaking_v2 import install


install()

from scripts.evaluate import run_c3_frozen_robust_external_v1 as implementation  # noqa: E402
from scripts.evaluate.c3_source_set_metrics_v2 import report_policy  # noqa: E402


implementation.report_policy = report_policy


if __name__ == "__main__":
    implementation.main()
