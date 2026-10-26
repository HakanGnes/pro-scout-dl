# Player-swap simulator — sanity checks

The simulator replaces a player by another player's season profile in every league match he played and lets the passing-network models (DeepSets + GNN, 10 networks) predict the team's non-penalty xG before and after. These checks have pass criteria fixed in `sim/checks.py` before they were run.

| check | result | summary |
| --- | --- | --- |
| Self-swap | pass | 294 players: mean change -0.0076, mean |change| 0.0112, 90th percentile |change| 0.0222 npxG per match (pass if mean |change| < 0.05) |
| Forward → goalkeeper | pass | 98 teams, each forward replaced by the most-used keeper from another team: change negative for 100% of teams, median change -0.104 npxG per match (pass if ≥ 90%) |
| Candidate ranking | pass | Barcelona: replacing L. Messi with each of 160 forwards (≥ 1800 league minutes); Spearman(box-entry rate, change) = +0.94 (pass if > 0) |

## Self-swap

294 players: mean change -0.0076, mean |change| 0.0112, 90th percentile |change| 0.0222 npxG per match (pass if mean |change| < 0.05)

| team | player | matches | delta | abs_delta |
| --- | --- | --- | --- | --- |
| Barcelona | L. Messi | 36 | +0.073 | 0.073 |
| Valencia | Neto | 33 | -0.061 | 0.061 |
| Real Madrid | K. Navas | 27 | -0.054 | 0.054 |
| PSG | A. Aréola | 34 | -0.046 | 0.046 |
| Atlético Madrid | J. Oblak | 37 | -0.046 | 0.046 |
| Deportivo Alavés | Fernando Pacheco | 38 | -0.039 | 0.039 |
| Nantes | C. Tătărușanu | 37 | -0.037 | 0.037 |
| Sassuolo | A. Consigli | 37 | -0.035 | 0.035 |
| Manchester City | N. Otamendi | 34 | -0.034 | 0.034 |
| Manchester United | David de Gea | 37 | -0.032 | 0.032 |

## Forward → goalkeeper

98 teams, each forward replaced by the most-used keeper from another team: change negative for 100% of teams, median change -0.104 npxG per match (pass if ≥ 90%)

| team | forward | keeper | matches | before | delta |
| --- | --- | --- | --- | --- | --- |
| Girona | C. Stuani | M. Ryan | 33 | 1.030 | -0.005 |
| Stoke City | M. Diouf | M. Ryan | 35 | 0.951 | -0.012 |
| Brighton & Hove Albion | G. Murray | J. Lössl | 35 | 0.986 | -0.012 |
| Burnley | A. Barnes | M. Ryan | 36 | 0.899 | -0.015 |
| Crystal Palace | C. Benteke | M. Ryan | 31 | 1.103 | -0.015 |
| Wolfsburg | D. Origi | M. Ryan | 31 | 0.967 | -0.015 |
| Real Betis | Sergio León | M. Ryan | 31 | 1.044 | -0.019 |
| Southampton | S. Long | M. Ryan | 30 | 1.024 | -0.019 |
| Leganés | C. Beauvue | M. Ryan | 28 | 0.943 | -0.020 |
| Mainz 05 | Y. Muto | M. Ryan | 27 | 0.866 | -0.025 |

## Candidate ranking

Barcelona: replacing L. Messi with each of 160 forwards (≥ 1800 league minutes); Spearman(box-entry rate, change) = +0.94 (pass if > 0)

| player_in | name | matches | skipped | delta | delta_per_90 | se_matches | sd_models | box_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25707 | E. Hazard | 36 | 0 | -0.040 | -0.043 | 0.037 | 0.017 | 3.718 |
| 3361 | A. Sánchez | 36 | 0 | -0.085 | -0.092 | 0.037 | 0.019 | 4.278 |
| 237034 | Gonçalo Guedes | 36 | 0 | -0.093 | -0.101 | 0.036 | 0.019 | 3.446 |
| 11063 | Suso | 36 | 0 | -0.094 | -0.102 | 0.037 | 0.026 | 3.795 |
| 20941 | A. Ljajić | 36 | 0 | -0.122 | -0.132 | 0.036 | 0.020 | 2.907 |
| 5268 | Sergio León | 36 | 0 | -0.495 | -0.534 | 0.042 | 0.054 | 0.455 |
| 94764 | Y. Muto | 36 | 0 | -0.502 | -0.542 | 0.042 | 0.055 | 0.381 |
| 20661 | A. Paloschi | 36 | 0 | -0.513 | -0.554 | 0.043 | 0.054 | 0.329 |
| 38377 | L. Depoitre | 36 | 0 | -0.526 | -0.568 | 0.043 | 0.059 | 0.148 |
| 3714 | C. Stuani | 36 | 0 | -0.530 | -0.572 | 0.043 | 0.059 | 0.235 |

## What the numbers mean

- `delta`: change in the team's predicted non-penalty xG per match, averaged over the matches the replaced player played and over all networks.
- A candidate's profile comes from his own team; a player who thrives in a dominant side carries that context with him. Treat results as a simulation under the model's assumptions.
