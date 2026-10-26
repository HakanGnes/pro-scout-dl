# Pre-registered experiment: richer edge features

Protocol: [`docs/preregistration/2026-10-26-gnn-edge-features.md`](../../docs/preregistration/2026-10-26-gnn-edge-features.md), registered before this run. Run on 2026-10-26; target xG model `lgbm@2026-10-26T13:08:31+00:00`.

## Primary outcome

GNN-v2 − DeepSets absolute error on the league test weeks: **-0.0045**, 95% interval [-0.0132, +0.0045] (2,000 bootstrap resamples of matches).

Success criterion (upper end below 0): **not met**. Richer edge features do not make the GNN beat DeepSets; the Phase 1 conclusion stands.

## Secondary outcomes (not used to decide)

- GNN-v2 − DeepSets, World Cup + Euro: +0.0058 [-0.0134, +0.0243]
- GNN-v2 − Phase 1 GNN, league test weeks: -0.0027 [-0.0118, +0.0068]

## Test metrics

| model | MAE | RMSE | R² | Spearman |
| --- | --- | --- | --- | --- |
| DeepSets (Phase 1) | 0.468 | 0.608 | 0.365 | 0.582 |
| GNN (Phase 1) | 0.466 | 0.609 | 0.363 | 0.583 |
| GNN-v2 | 0.463 | 0.604 | 0.374 | 0.586 |

World Cup + Euro:

| model | MAE | RMSE | R² | Spearman |
| --- | --- | --- | --- | --- |
| DeepSets (Phase 1) | 0.411 | 0.542 | 0.371 | 0.685 |
| GNN (Phase 1) | 0.418 | 0.552 | 0.348 | 0.671 |
| GNN-v2 | 0.416 | 0.537 | 0.384 | 0.700 |

GNN-v2 seeds: seed 0 best epoch 35 (val MSE 0.3263), seed 1 best epoch 43 (val MSE 0.3184), seed 2 best epoch 31 (val MSE 0.3186), seed 3 best epoch 35 (val MSE 0.3246), seed 4 best epoch 38 (val MSE 0.3195).
