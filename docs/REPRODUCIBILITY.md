# Reproducibility guide

## Frozen contract

1. Build train, calibration, and test artifacts before loading the evaluated checkpoint.
2. Record SHA-256 for every protected input and generated queue.
3. Use predefined seeds 42, 43, and 44.
4. Never tune prompts, thresholds, donor assignment, or structured decoding on a reported external test set.

## Required local assets

The repository excludes licensed data and weights. Users obtain them under their original terms and place them under:

```text
data/raw/averitec/
data/raw/averimatec/
data/raw/finfact/
models/BAAI/bge-reranker-v2-m3/
models/Qwen/Qwen3-Reranker-0.6B/
models/cross-encoder/ms-marco-MiniLM-L6-v2/
models/mixedbread-ai/mxbai-rerank-large-v1/
```

Manifests state the exact input hashes. A mismatch means the reported metric is not reproduced.

## Core pipeline

1. Repair legacy Source-Swap into the publisher-visible URL contract.
2. Remove query/document leakage and freeze train, calibration, clean blind, AVerImaTeC, and FinFact artifacts.
3. Train the multitask publisher router for three seeds.
4. Evaluate each checkpoint once in FP32 on clean blind and external splits.
5. Run frozen strong rerankers with the same publisher-visible input view and label-independent tie rule.
6. Select C3 only on predefined calibration domains, freeze the policy, then evaluate external data once.
7. Run C4 with the same generator contract and stable candidate ordering for raw/C2/C3.

## Result integrity

Small publication-safe summaries are under `results/release/`. Large predictions remain local. A shared archive should include command/configuration, code commit, input/output hashes, seed and precision, model identifier and hashes, frozen-policy ID, uncertainty, and a `supersedes` field when appropriate.

## Secrets

Set API credentials in the shell or a private `.env`. Never paste a key into scripts, notebooks, Word files, result JSON, or Git commits. `.env.example` contains only a public endpoint template.
