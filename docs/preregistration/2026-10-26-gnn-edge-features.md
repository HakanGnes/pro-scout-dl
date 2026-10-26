# Pre-registered experiment: richer edge features for the passing-network GNN

- **Registered:** 2026-10-26, before running it. The commit that adds this file is the timestamp.
- **Status:** run once on 2026-10-26, after this file was committed. Result: **not supported** — GNN-v2 − DeepSets
  −0.0045 MAE, 95% interval [−0.0132, +0.0045]. Full result:
  [`reports/pass_network_gnn/prereg_edge_features.md`](../../reports/pass_network_gnn/prereg_edge_features.md).

## Background

In Phase 1 the GNN (players + who passes to whom) and DeepSets (players only, no edges) tied on
team non-penalty xG: GNN − DeepSets MAE −0.0018, 95% interval [−0.0081, +0.0046] on the league test
weeks (`reports/pass_network_gnn/README.md`). The edges carried only two features: pass count and
mean progression. This experiment gives the network structure one fair, fixed second chance.

## Hypothesis

A GNN whose edges also describe *where* and *what kind of* passes form each link beats DeepSets.

## The one change

Edge features go from 2 to 6, computed from the same completed passes that define the edge:

| feature | definition |
| --- | --- |
| `log_passes` | log(1 + completed passes on the edge) — unchanged |
| `mean_progress_10m` | mean progression towards goal, in units of 10 m — unchanged |
| `share_final_third` | share of the edge's passes ending in the final third (x ≥ 66.7) |
| `share_box` | share of the edge's passes ending in the penalty box |
| `mean_start_x` | mean start x of the edge's passes (0–1) |
| `mean_start_y` | mean start y of the edge's passes (0–1) |

Everything else stays exactly as in Phase 1: node features, `PassGNN` architecture (3 GINE layers,
hidden 64, dropout 0.1), Adam lr 1e-3, weight decay 1e-4, batch 64, early stopping on validation MSE
with patience 30, seeds 0–4, the same dataset split, the same xG model for the target. DeepSets is
not retrained differently: it does not read edges, so its Phase 1 predictions are the comparison.

## Primary outcome

Paired difference in absolute error, GNN-v2 (5-seed ensemble) − DeepSets (5-seed ensemble), on the
league test weeks, 95% interval from 2,000 bootstrap resamples of matches (`paired_bootstrap`, seed 0).

**Success:** the upper end of that interval is below 0.

## Secondary outcomes (reported, not used to decide)

- The same difference on World Cup + Euro.
- GNN-v2 − Phase 1 GNN on the league test weeks.

## Rules

- One run. No tuning of any hyperparameter, feature definition or seed set after seeing results.
- The result is reported in `reports/pass_network_gnn/` and the main README whatever it is.
- Any further variant (other architectures such as GATv2, other targets) needs a new pre-registration
  file committed before it is run.
