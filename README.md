# ProvTimeRAG

ProvTimeRAG studies provenance-aware routing for time-sensitive retrieval-augmented generation (RAG). The central problem is not only whether a passage is relevant, but whether its **observable publisher identity**, document/version state, temporal validity, and support status match an atomic claim.

This repository accompanies an anonymous WSDM 2027 submission draft. It contains final training and evaluation code, leakage-controlled Source-Swap construction, frozen external evaluations, and reproducibility metadata. Raw datasets, model weights, API credentials, and large prediction files are intentionally excluded.

## Main idea

ProvTimeRAG decomposes provenance control into four layers:

1. **C1 — metadata recovery and adjudication.** Recover observable publisher/actor metadata and record uncertainty using a human-audited schema.
2. **C2 — provenance-state router.** A multitask cross-encoder routes each atomic claim to the matching publisher/document state, while retaining temporal/version and insufficiency behavior.
3. **C3 — conservative structured decoding.** A frozen bundle-level policy makes small globally consistent corrections without tuning on the external test set.
4. **C4 — generation contract.** A fixed generator consumes routed evidence and is evaluated for answer quality, citation precision/recall, latency, and abstention.

## Headline results

- Leakage-controlled clean blind publisher routing: **88.81% ± 0.60% Top-1** over three predefined seeds.
- FinFact cross-dataset publisher routing: **93.83% ± 0.59% Top-1** over 11,164 groups.
- Strongest frozen off-the-shelf baseline, Qwen3-Reranker-0.6B, reaches 87.93% on FinFact; the proposed router improves by about **5.89 percentage points**.
- Removing URL/domain evidence lowers mean Top-1 by 17.75 points on development, 20.17 points on clean blind, and 15.16 points on FinFact.
- On the fair-order end-to-end set (678 API-eligible examples), structured routing improves citation precision from 0.500 to 0.547 and recall from 0.892 to 0.940, with essentially unchanged answer F1 within the paired interval.

Exact frozen numbers, caveats, and superseded experiments are recorded in [`results/release/main_results.json`](results/release/main_results.json) and [`docs/RESULTS_CARD.md`](docs/RESULTS_CARD.md).

## Repository map

```text
src/provtimerag/           Core data models, routing, metrics, and utilities
scripts/prepare/           Leakage-controlled dataset construction and repair
scripts/train/             Final multitask publisher router and ablations
scripts/evaluate/          Frozen holdout, strong baseline, C3, and C4 evaluation
tests/                     Contract and regression tests
configs/                   Reproducible experiment configuration
docs/                      Research contract, results card, and execution notes
results/release/           Publication-safe result summaries only
paper/                     Chinese drafting artifact and migration guide
```

## Installation

Python 3.11 or 3.12 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml,dev]'
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

## Reproduction

The public repository does not redistribute datasets or model weights. Place licensed inputs at the paths described in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), verify the frozen SHA-256 values, and run the documented commands. Never tune a model or decoder on a frozen holdout.

```bash
PYTHONPATH=src python -m pytest -q
```

## Reporting policy

- Report predefined seeds 42, 43, and 44 as mean ± sample standard deviation.
- Treat repeated predictions across seeds as dependent; paired significance is computed per seed.
- Do not claim that C3 significantly improves external bundle accuracy: its confidence intervals include zero.
- Do not claim that multitask training improves publisher routing over a compute-matched source-only model. Its value is consolidating publisher, temporal/version, and insufficiency behavior in one router.
- Legacy v1 Source-Swap results and gold-first generation queues are superseded and must not be used for paper claims.

## Privacy and security

API keys are read from environment variables only. `.env`, raw/interim/processed data, weights, checkpoints, API generations, human-annotation spreadsheets, and local QA artifacts are ignored by Git.

## License

Code is released under the Apache License 2.0. Dataset and model licenses remain with their original providers.
