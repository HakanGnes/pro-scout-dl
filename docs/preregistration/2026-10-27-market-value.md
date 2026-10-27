# Pre-registered study: 12-month market value — GBM vs. LSTM vs. Transformer

- **Registered:** 2026-10-27, before any model in this study was trained. This file is committed
  together with the code that runs the study; that commit is the timestamp, and the run uses the code
  at that commit.
- **Status:** run once on 2026-10-27 at commit `bf45e37`, after this file was committed with the code. Result:
  **supported**: Transformer − LightGBM −0.0060 MAE, 95% interval [−0.0076, −0.0044]. Full result:
  [`reports/market_value/README.md`](../../reports/market_value/README.md).
- **Disclosure:** while exploring the data I computed, for every anchor date (test ones included),
  the number of samples, the mean and standard deviation of the target and the share of unchanged
  values. These need no fitting and do not depend on any model choice. Training speed was measured on
  targets replaced by random noise, and the pipeline was tested end to end on synthetic data only.

## Question

Given everything known about a player on a date *t*, how well can we predict the market value
Transfermarkt will show for him one year later? And does a deep sequence model reading the raw
monthly history (LSTM, Transformer) beat gradient boosting on hand-made summaries of the same history?

## Data

Transfermarkt open dataset (dcaribou/transfermarkt-datasets, CC0), snapshot frozen by its maintainers
in July 2026 (valuations up to 2026-06-12). SHA-256 of the files used:

| file | sha256 |
| --- | --- |
| player_valuations.csv.gz | `608e32e5ad2231074f3505c469ad727be66d7e6efda6af918afd85379439631d` |
| players.csv.gz | `d22e407981d5b51a79bf8ff59835729f3526f7dc3495a6d8ed2f852ed2e86403` |
| appearances.csv.gz | `e096c2b158fc3d52c4550526c4d4850dd62f31007652f4bcae112eadb1646802` |
| transfers.csv.gz | `90326983daf7e6ac7aabdfe62b90936d9bf1dd2171e53dec75c3751eb1620a83` |
| competitions.csv.gz | `8924ddfbc0e9989f4a42a3c32ebb6faa671614625086d7fa84353f955352ea88` |

Appearances cover 14 European first divisions (England, Spain, Italy, Germany, France, Portugal,
Netherlands, Belgium, Türkiye, Russia, Greece, Scotland, Ukraine, Denmark) from 2012/13, plus cups,
European and national-team competitions.

## Samples

- **Anchor dates *t*:** 1 July and 1 January, from 2013-07-01 to 2025-01-01 (24 anchors). Transfermarkt
  publishes most revaluations in June and December, so each anchor sits just after a round.
- **Population at *t*:** players with at least one appearance (any minutes) in one of the 14 leagues in
  the 12 calendar months before *t*, a valuation dated in the 365 days before *t*, a known date of
  birth and an age at *t* between 16 and 40.
- **Current value** *V(t)*: the last valuation dated before *t*.
- **Target value** *V(t+12)*: the last valuation dated in [*t*, *t* + 365 days). Players with no
  valuation in that year (retired, left the data) are dropped; the dropped share is reported per anchor.
- **Target:** *y* = log *V(t+12)* − log *V(t)*. A prediction ŷ maps back to a value *V(t)*·exp(ŷ).

## Point-in-time rule

Every input is computed from records dated before *t*. Fields that describe the player *as of the
snapshot* are never used: current club, contract expiry, highest market value, international caps
and goals, agent. Transfers dated in the future (loan returns, pre-agreed moves) are ignored until
their date. `python -m models.market_value run` refuses a dataset not built from the files above.

## Splits (by anchor date; target windows never reach into a later split)

| split | anchors | target windows end by |
| --- | --- | --- |
| train | 2013-07-01 … 2020-07-01 (15) | 2021-07-01 |
| validation | 2021-07-01, 2022-01-01 | 2023-01-01 |
| test | 2023-01-01 … 2025-01-01 (5) | 2026-01-01 |

Anchors 2021-01-01 and 2022-07-01 are not used, so that no training target overlaps validation and no
validation target overlaps test.

## Inputs

All models are built from one monthly table per sample: the 36 calendar months before *t*. Per month
(14 channels): log value as of month end (carried forward), whether a value exists yet, the same
value relative to *V(t)*, whether a revaluation happened, league minutes/90, league appearances,
goals and assists (all competitions), yellow + red cards, European-cup minutes/90, domestic-cup
minutes/90, a transfer flag, log(1 + fee in €m), and the strength of the league played most in that
month (median log *V(t)* of players with league minutes in the 12 months before *t*).
National-team minutes are not used: the snapshot has national-team appearances only from December
2025, after every anchor.

Static inputs: age at *t*, position and sub-position, foot, height (150–215 cm, otherwise missing),
league at *t* (most recent league appearance) and its strength, anchor month (January/July), and a
market-momentum index (mean 12-month log change, clipped to ±2, of the values known at *t* of all
players active at *t*).

- **LightGBM** gets the static inputs plus fixed summaries of the monthly table (value changes over
  6/12/24/35 months, months since the last revaluation, revaluation counts, peak value in 36 months,
  minutes/apps/goals/assists/cup/European/national minutes over the last 6, last 12 and the 12 before,
  per-90 rates over 12 months, transfer recency, count and last fee, league strength now and 12 months
  ago). The list is fixed in code before training and not changed after.
- **LSTM** and **Transformer** get the static inputs and the raw 36-month sequence.

## Models

1. **Persistence:** ŷ = 0 (the value does not change).
2. **Age curve:** the training-set median of *y* per (age in whole years, position); cells with fewer
   than 20 training samples use the median for that age, then the overall median.
3. **LightGBM:** L1 objective, learning rate 0.03, 63 leaves, min 50 samples per leaf, 0.8 feature and
   bagging fractions, up to 5,000 rounds with early stopping (200 rounds) on validation MAE.
4. **LSTM:** 2 layers, hidden 64, dropout 0.1; last hidden state concatenated with a static MLP, then a
   2-layer head.
5. **Transformer encoder:** d_model 64, 4 heads, 2 layers, feed-forward 128, dropout 0.1, learned
   positional embeddings, months before the player's first valuation masked; last token and mean
   pooling concatenated with the static MLP, then a 2-layer head.

Neural models: L1 loss, AdamW (lr 1e-3, weight decay 1e-4), batch 256, gradient norm clipped at 1.0,
up to 100 epochs, early stopping on validation MAE with patience 10 (best epoch kept), inputs
standardised with training statistics. Models 3–5 are each
trained with seeds 0–4 and their 5 predictions averaged. No hyperparameter search.

## Outcomes

Metric: mean absolute error of *y* on the test anchors (MAE of log value; 0.10 ≈ 10% off).
Uncertainty: 95% intervals from 2,000 bootstrap resamples of **players** (a player appears at several
anchors), seed 0.

- **Primary:** paired difference Transformer − LightGBM. **Success:** upper end of the interval below 0.
- **Secondary (reported, not used to decide):** LSTM − LightGBM; Transformer − LSTM; every model −
  persistence; median absolute percentage error of value; MAE by value band, age band, position and
  league; MAE on samples whose value did and did not change.

## Rules

- One run of the code at the registering commit. No tuning after seeing test results.
- Results go in `reports/market_value/` and the main README whatever they are.
- If the Transformer does not beat LightGBM, LightGBM is the model used downstream (app, fusion score).
- Any further variant (other horizons, more inputs, other architectures) needs a new pre-registration.
