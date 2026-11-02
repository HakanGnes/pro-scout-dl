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

Validation AUC, fitted settings and per-paper, new-pair and target-league tables: `results.json`, `auc_by_*.csv`.

## Secondary tables (test AUC)

| paper | claims | came true | metadata | TF-IDF | BERTurk | fusion |
| --- | --- | --- | --- | --- | --- | --- |
| aspor | 636 | 22.6% | 0.748 | 0.662 | 0.698 | 0.775 |
| fanatik | 392 | 29.1% | 0.764 | 0.594 | 0.647 | 0.782 |
| fotomac | 871 | 18.1% | 0.779 | 0.689 | 0.682 | 0.792 |
| hurriyet_spor | 317 | 22.7% | 0.766 | 0.619 | 0.672 | 0.782 |
| milliyet_skorer | 356 | 29.2% | 0.757 | 0.681 | 0.704 | 0.788 |
| ntvspor | 292 | 18.5% | 0.767 | 0.621 | 0.616 | 0.783 |
| sabah_spor | 57 | 35.1% | 0.707 | 0.699 | 0.708 | 0.714 |
| sporx | 241 | 30.3% | 0.739 | 0.691 | 0.688 | 0.772 |

| group | claims | came true | metadata | TF-IDF | BERTurk | fusion |
| --- | --- | --- | --- | --- | --- | --- |
| pair new in the news (no article in the year before) | 887 | 10.7% | 0.698 | 0.652 | 0.666 | 0.737 |
| pair in the news before | 2,275 | 28.3% | 0.750 | 0.633 | 0.655 | 0.764 |
| target is a Turkish club | 2,739 | 23.3% | 0.787 | 0.653 | 0.675 | 0.797 |
| target is a foreign club | 423 | 23.6% | 0.635 | 0.662 | 0.655 | 0.682 |

The fusion model's probabilities (`reliability_fusion.csv`, ten equal groups of test claims) follow the
observed rates from about 20% upwards (e.g. predicted 30% → 36% came true, 67% → 65%) but are too high below
that (predicted 8% → 2.5%, 17% → 12%).

**Correction (2026-11-02, after the run):** in the first version of `auc_by_site.csv` and of
`data/processed/rumours/predictions.csv.gz`, the *site* model's column overwrote the paper column, so
paper names were lost. The tables above were rebuilt from the saved predictions with the paper taken from
each URL; no model was retrained and no registered number changed. The code now names prediction
columns `p_<model>` and group columns `auc_<model>`.

## Reading the result

- **The wording adds a little, and only a little.** Adding BERTurk to the metadata model raises test AUC by
  0.015; the interval's lower end is just above zero. Almost all of what can be predicted comes from what is
  known without reading the headline: how many papers ran the pair in the last month, how long the pair has
  been in the news, the player's market value and age.
- **BERTurk reads the headlines better than a bag of words**, by a small margin (+0.019 AUC alone, +0.010
  within the fusion), and only the BERTurk fusion beats metadata alone; the TF-IDF fusion does not.
- **The paper's name alone says almost nothing** (AUC 0.535). Papers differ in how often their rumours come
  true (18%–35% in 2025), but within each paper the models rank rumours about as well as overall.
- **Foreign targets are harder** (fusion AUC 0.68 against 0.80 for Turkish clubs): the metadata knows
  little about how Turkish papers cover foreign clubs.
- **Limits.** One claim in three in the study is not a clean rumour about the right player and target club
  ([`checks.md`](checks.md)); this lowers every AUC. Labels come from Transfermarkt's transfer records, which
  grow over the years (11k moves in 2018, 17k in 2024), so older "did not happen" labels are noisier. The
  text is the URL slug, not the full headline.

## Exploratory analyses (after the run, not registered)

**Players with no transfer record before the article.** The metadata input "current club known" has the
second largest gain in the GBM. Claims on players with no Transfermarkt transfer before the article
never came true in any period (train 0 of 4,106, validation 0 of 198, test 0 of 167): these are mostly
namesakes and players whose moves Transfermarkt does not record, so the input partly measures the label
source's coverage. Dropping these 167 test claims leaves the result unchanged: AUC(fusion) − AUC(metadata)
+0.0179 [+0.0021, +0.0386], BERTurk − TF-IDF +0.0163 [−0.0013, +0.0345] (2,995 claims, same bootstrap).

**What the wording says** (word 1–2-grams, logistic regression on the training period only; placeholders
as in the study). Toward *comes true*: "istanbula", "imzaya geliyor", "bitiriyor", "görüşmeler/görüşmelere",
"anlaştı", "teklifi belli", "geliyor", "bomba gelişme". Toward *does not*: "ilgileniyor", "sürprizi",
"harekatı", "hamlesi", "rakip", "eski", "ayrılık", "menajeri", "istiyordu", and a second club in the same
headline. Concrete, late-stage language (travel, signing, talks under way) separates from early interest
and competition. A few first names survived the masking ("josef", "larsen", "murat"); they are rare.
