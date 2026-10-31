# Transfer rumour credibility: results

Pre-registration: [`docs/preregistration/2026-10-29-rumour-credibility.md`](../../docs/preregistration/2026-10-29-rumour-credibility.md). Run at commit `4acbe6f` on 2026-10-31T14:45:39+00:00.

Test period 2025: 3,162 rumour claims, 1,109 player-club pairs, 23.4% came true within 180 days.

## Primary outcome

AUC(fusion) - AUC(metadata) = +0.0154 [+0.0009, +0.0339] (95% interval, 2,000 bootstrap resamples of player-club pairs). **Supported.**

## All models on the test period (probabilities recalibrated on validation)

| model | AUC | avg precision | Brier | log loss | mean p |
| --- | --- | --- | --- | --- | --- |
| base | 0.5000 | 0.2337 | 0.1797 | 0.5454 | 0.259 |
| site | 0.5347 | 0.2429 | 0.1795 | 0.5447 | 0.258 |
| metadata | 0.7675 | 0.5043 | 0.1502 | 0.4669 | 0.250 |
| tfidf | 0.6537 | 0.4001 | 0.1674 | 0.5136 | 0.253 |
| bert | 0.6726 | 0.4241 | 0.1644 | 0.5058 | 0.252 |
| fusion | 0.7829 | 0.5519 | 0.1440 | 0.4499 | 0.247 |
| fusion_tfidf | 0.7735 | 0.5439 | 0.1456 | 0.4540 | 0.247 |

## Secondary comparisons (AUC difference, 95% interval)

- bert - tfidf: +0.0189 [+0.0013, +0.0360]
- bert - metadata: -0.0949 [-0.1434, -0.0346]
- tfidf - metadata: -0.1138 [-0.1645, -0.0518]
- fusion_tfidf - metadata: +0.0059 [-0.0090, +0.0239]
- fusion - fusion_tfidf: +0.0095 [+0.0025, +0.0165]
- metadata - site: +0.2328 [+0.1653, +0.2873]
- tfidf - site: +0.1190 [+0.0824, +0.1533]

Validation AUC, fitted settings and per-paper / per-month tables: `results.json`, `auc_by_*.csv`.
