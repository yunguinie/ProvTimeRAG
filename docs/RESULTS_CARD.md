# ProvTimeRAG frozen results card

Updated: 2026-08-10. This file separates paper-eligible results from diagnostic or superseded runs.

## Primary claim: publisher-aware provenance routing (C2)

All values are frozen before external evaluation. The clean splits remove query overlap and nearly all document overlap; source/domain overlap is reported separately rather than hidden.

| Evaluation | Groups | Proposed C2 Top-1 | Notes |
|---|---:|---:|---|
| Leakage-controlled development | 460 | 0.8638 ± 0.0082 | seeds 42/43/44 |
| Clean blind holdout | 828 | 0.8881 ± 0.0060 | unseen publishers: 0.8929 ± 0.0112 |
| FinFact external | 11,164 | 0.9383 ± 0.0059 | 2,365 bundles, 2,850 publishers |

### Strong frozen baselines

| Method | Clean blind Top-1 | FinFact Top-1 | FinFact bundle exact |
|---|---:|---:|---:|
| MS MARCO MiniLM-L6-v2 | 0.5399 | 0.5206 | 0.1167 |
| BGE reranker v2 m3 | 0.5990 | 0.5962 | 0.1852 |
| mxbai-rerank-large-v1 | 0.6087 | 0.6745 | 0.2837 |
| Qwen3-Reranker-0.6B | 0.7343 | 0.8793 | 0.5979 |
| ProvTimeRAG C2 (3-seed mean) | **0.8881** | **0.9383** | **0.7686** |

The paired C2-versus-Qwen3 comparisons are significant for each predefined seed on both clean blind and FinFact; do not pool seeds as independent observations.

## What creates the gain?

| Split | Full C2 | No-URL multitask | Difference |
|---|---:|---:|---:|
| Development | 0.8638 | 0.6862 | +17.75 pp |
| Clean blind | 0.8881 | 0.6864 | +20.17 pp |
| FinFact | 0.9383 | 0.7866 | +15.16 pp |

A compute-matched source-only model can slightly exceed C2 on publisher-only blind accuracy. The paper claim is therefore not “multitask always improves publisher accuracy.” The defensible result is that a single multitask router preserves nearly all source-only publisher quality while also solving temporal/version routing and insufficiency detection, where source-only training collapses.

## Conservative structured decoding (C3)

On FinFact, the frozen robust decoder raises bundle exact from 0.7686 to 0.7721 (mean +0.35 pp). Per-seed McNemar p-values are 0.233, 0.280, and 0.280; all bootstrap intervals cross zero. Binary multi-source error rises slightly. C3 is therefore a secondary safety/consistency mechanism, not a significant external SOTA claim.

## End-to-end citation behavior (C4)

Fair-order, same-candidate, same-generator evaluation on 678 API-eligible examples:

| Method | Answer F1 | Citation P | Citation R | Citation hit |
|---|---:|---:|---:|---:|
| Raw | 0.6192 | 0.4998 | 0.8916 | 0.9071 |
| C2 | 0.6162 | 0.5302 | 0.9028 | 0.9189 |
| C3 | 0.6143 | **0.5470** | **0.9400** | **0.9543** |

C3 citation improvements have paired bootstrap intervals entirely above zero. Answer-F1 differences are small and their intervals include zero. The correct statement is improved attribution/citation quality without a statistically supported answer-quality gain.

## Metadata recovery (C1)

The 50-row independently adjudicated set gives row exact match 0.48 for the best audited prompt variant. Field accuracies vary substantially. C1 supports the feasibility of recovering observable metadata and building an audit trail; it is not presented as a standalone high-accuracy classifier.

## Superseded results

- Legacy Source-Swap v1 exposed an incorrect URL contract and must not support URL-aware claims.
- AVerImaTeC donor v1 collapsed nearly every negative onto one domain; v2 balances positive and negative publisher histograms and supersedes it.
- Gold-first or non-canonical C4 queues are diagnostic only. Only stable hash order/fair-order queues support the end-to-end paper result.
- Early C3 bundle exact values around 0.36–0.41 were produced before publisher-identity and metric repairs. They remain useful for audit history but are not final headline numbers.
