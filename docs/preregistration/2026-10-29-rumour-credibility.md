# Pre-registered study: do the words of a transfer rumour tell whether it comes true?

- **Registered:** 2026-10-29, before any model in this study was trained on real data. This file is
  committed together with the code that runs the study; that commit is the timestamp, and the run uses
  the code at that commit.
- **Status:** registered, not yet run. The run waits for the archived pages of the study's articles
  (publication dates, see *Dates*).
- **Disclosure — what was seen before registering:**
  - Per year and stance, the share of claims that came true, for every year 2011–2026, with an
    earlier 365-day horizon (2025 rumours: 23%). With the registered 180-day horizon, the shares for
    years up to 2024 only; from 2025 on the code hides them.
  - Rumour share that came true in train (13.2%) and validation (25.7%), and per paper in train.
  - The lag from headline to move for rumours that came true, over all years 2018–2026 test included
    (median 20 days; 90% within 182 days), which set the 180-day horizon.
  - The number of claims, articles, player–club pairs and players in the test period (not their outcomes).
  - Transfers per year in the Transfermarkt snapshot.
  - The stance rules (rumour / announcement / other) were written and then revised on 162 hand-labelled
    headlines from 2018–2024 (see *Checks*). No 2025 headline was used to write any rule.
  - The study code was run end to end on synthetic data only.

## Question

Turkish sports papers publish many transfer rumours ("X, Y'nin radarında", "Z için teklif"). Given a
rumour that player *p* will move to club *c*, published on day *t*: does the wording of the headline
say anything about whether the move happens, **beyond** what is known without reading it (which paper
ran it, how much the pair was in the news, who the player is)? And does a fine-tuned Turkish BERT
read that wording better than a bag of words?

## Data

- **Headlines:** the Internet Archive index of eight Turkish sports sites (fotomac, fanatik, aspor,
  sporx, hurriyet_spor, milliyet_skorer, ntvspor, sabah_spor), walked by `python -m etl history-index`
  and finished on 2026-10-29: 324,561 candidate transfer URLs, 286,132 articles once copies of the same
  slug on the same site are merged. The text used is the URL slug (lower case, ASCII-folded), with the
  sites' numeric article ids removed.
- **Outcomes:** Transfermarkt open dataset (CC0), the snapshot already registered for Phase 3
  (frozen July 2026).

SHA-256 of the inputs (checked by `python -m models.rumours run`):

| file | sha256 |
| --- | --- |
| history/aspor/index.jsonl.gz | `381297bbf064e308d55a7ebe0927cb1c90933226bd846f6dd446f0cd80598770` |
| history/fanatik/index.jsonl.gz | `e382c02d9158ed8d9fae6584f1ba8c63a397de5e6c99fa5c4694d37fefa12893` |
| history/fotomac/index.jsonl.gz | `20f08cdb129e7140dd71e4448a380ebee524e65ccbf82559e18a25d14567abfa` |
| history/hurriyet_spor/index.jsonl.gz | `184f57a48ad27a699bbc21eb9225f855aebb2864475aee7fabf4062a9b2448db` |
| history/milliyet_skorer/index.jsonl.gz | `78f9633a3a7abdc438605aeebb544b78924e56207b7c6444a66dd598e1f305bb` |
| history/ntvspor/index.jsonl.gz | `f01f0d1f7039a784442c1823dff65b7162e2f346655ea5a8f415fac0f107792c` |
| history/sabah_spor/index.jsonl.gz | `1d6b376293770ca6a237bb42623b19291f1aeda816a3f34eb20c2e90742e71a7` |
| history/sporx/index.jsonl.gz | `2003c3a8976f23c77b3a85819581116666fb0180e5c1ca3928c96bd21b31e101` |
| transfermarkt/players.csv.gz | `d22e407981d5b51a79bf8ff59835729f3526f7dc3495a6d8ed2f852ed2e86403` |
| transfermarkt/clubs.csv.gz | `aa57ab56e87faffaea1e8697439a32caf10116d8f2f2dfb3a57193c0bb1d641b` |
| transfermarkt/transfers.csv.gz | `90326983daf7e6ac7aabdfe62b90936d9bf1dd2171e53dec75c3751eb1620a83` |
| transfermarkt/player_valuations.csv.gz | `608e32e5ad2231074f3505c469ad727be66d7e6efda6af918afd85379439631d` |

The archived pages (`history/titles.jsonl.gz`) are fetched after registration by the registered code;
their hash is recorded in the results.

## Dates

An article's date is the publication time its archived page states about itself (meta
`article:published_time` / `datePublished`), else the date in its URL (fotomac, aspor, sabah), else
the day the archive first captured it. The capture can come months after publication, so pages are
fetched for every rumour article captured from 2018-01-01 to 2026-06-30
(`data/processed/rumours/title_urls.txt`, 22,032 articles). The run starts only when at least 95% of
them have been fetched (a failed fetch counts as fetched).

## Claims and labels

- **Linking** (`features/rumours/entities.py`): Transfermarkt players named in full, or by last name
  when exactly one player with that name is tied to a club in the same headline; clubs by their
  Transfermarkt names, Turkish case endings allowed, plus a fixed list of nicknames.
- **Claim:** (article, player, club) where the club is not the player's club the day before.
- **Stance** (`features/rumours/labels.py`, rules fixed at this commit): announcement, rumour or other.
  **Only rumours are studied.**
- **Label:** 1 if Transfermarkt records the player moving to that club from 30 days before the article
  to 180 days after it, else 0. (Transfermarkt often dates a move at the contract start, after the deal.)

## Splits (by article date; a 180-day label window never reaches into a later split)

| split | articles dated | rumour claims (dates before page fetch) |
| --- | --- | --- |
| train | 2018-01-01 … 2022-12-31 | ~15,300 |
| validation | 2023-07-01 … 2024-06-30 | ~2,800 |
| test | 2025-01-01 … 2025-12-31 | ~3,600 |

Claims dated 2023-01-01 … 2023-06-30 and 2024-07-01 … 2024-12-31 are not used. Counts will change a
little once publication dates replace capture dates.

## Inputs

- **Text:** the slug with the claimed player → `oyuncu`, the claimed club → `hedefkulup`, the player's
  current club → `mevcutkulup`, other linked players / clubs → `baskaoyuncu` / `baskakulup`, Turkish
  endings kept (`galatasaraydan` → `hedefkulupdan`). Models learn how papers write, not which names
  came true before.
- **Metadata** (all from records dated before the article's day): paper; month; articles on the same
  pair in the previous 30 and 365 days; papers that ran the pair in the previous 30 days; days since the
  pair's first mention within the previous year; articles on the player in the previous 30 days and the
  number of clubs linked to him then; his last market value before the article (log) and age; position;
  whether the claimed club and his current club play in the Turkish league (club league as in the
  snapshot); whether his current club is known.

## Models

1. **Base rate:** the validation share of rumours that came true.
2. **Paper record:** each paper's training share, shrunk to the overall share with 50 pseudo-claims.
3. **Metadata:** LightGBM on the metadata (binary objective, learning rate 0.03, 31 leaves, min 50
   per leaf, 0.8 feature and bagging fractions, up to 3,000 rounds, early stopping after 200 on
   validation AUC), seeds 0–4 averaged.
4. **TF-IDF:** word 1–2-grams (min df 2) and character 3–5-grams (min df 3), sublinear tf, logistic
   regression; C from {0.1, 0.3, 1, 3, 10} by validation AUC.
5. **BERTurk:** `dbmdz/bert-base-turkish-uncased` fine-tuned on the masked text: max 64 tokens, batch
   32, AdamW lr 2e-5, weight decay 0.01, 10% linear warm-up then linear decay, 4 epochs, gradient norm
   1.0, mixed precision on GPU; the epoch with the best validation AUC is kept; seeds 0–2 averaged.
6. **Fusion:** logistic regression of the label on logit(metadata) and logit(BERTurk), fitted on
   validation. **Fusion-TF-IDF:** the same with TF-IDF.

Every model's probabilities are recalibrated on validation (logistic on the logit) before Brier score
and log loss are computed. No other tuning.

## Outcomes

Metric: ROC AUC on the 2025 test rumours (ranking; unaffected by the drift in the base rate between
periods, which comes partly from Transfermarkt recording more moves in later years). Uncertainty: 95%
intervals from 2,000 bootstrap resamples of **player–club pairs** (one pair is often in many articles),
seed 0, paired.

- **Primary:** AUC(fusion) − AUC(metadata). **Success:** lower end of the interval above 0, i.e. the
  wording adds information beyond everything else known about the rumour.
- **Secondary (reported, not used to decide):** BERTurk − TF-IDF; BERTurk − metadata; TF-IDF −
  metadata; fusion-TF-IDF − metadata; fusion − fusion-TF-IDF; metadata − paper record; TF-IDF − paper
  record. Brier score, log loss and average precision of every model; AUC by paper, by new vs.
  already-reported pair, by Turkish vs. foreign target club; a reliability table of the fusion model.
- **Exploratory (labelled as such):** the same models on the real page titles (with Turkish characters)
  where fetched.

## Checks reported before the run

- Stance and linking on 100 fresh hand-labelled headlines (2018–2024, not used to write any rule):
  precision of the rumour stance, share of correct player and of correct target club.
  *Done 2026-10-31 on 237 such headlines: [`reports/rumours/checks.md`](../../reports/rumours/checks.md).*

## Rules

- One run of `python -m models.rumours run` at the registering commit (or a later commit that changes
  nothing the run uses, stated in the report). No tuning after seeing test results.
- Results go in `reports/rumours/` and the main README whatever they are.
- If the primary outcome fails, the downstream credibility score uses the metadata model alone.
- Other horizons, inputs, label definitions or models need a new pre-registration.
