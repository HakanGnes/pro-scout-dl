# Rumour study: checks of linking and stance before the run

Done on 2026-10-31, before `python -m models.rumours run`, as the pre-registration asks
([`docs/preregistration/2026-10-29-rumour-credibility.md`](../../docs/preregistration/2026-10-29-rumour-credibility.md)).

## Sample

400 headlines drawn from the claims of five sites (aspor, fanatik, fotomac, hurriyet_spor,
milliyet_skorer), dated 2018–2024, stratified by the first automatic stance (160 rumour, 120
announcement, 120 other) and shuffled. One person labelled, for each headline: its type (rumour,
offer, agreed, official, other transfer news, not about a transfer), whether the linked player is
right, and which linked club or clubs are the clubs he is said to be going to. All labels are in
[`hand_labels.csv`](hand_labels.csv).

- **dev** (163 headlines, h001–h163): used to write and revise the stance rules.
- **check** (237 headlines, h164–h400): labelled after the rules were fixed and committed (`162f4ac`);
  no rule was changed after they were labelled. The figures below are on this set only.

For the comparison, *rumour* = rumour or offer, *announcement* = agreed or official, *other* = the rest.
Intervals are 95% Wilson intervals.

## Results on the check set

| check | result |
| --- | --- |
| Stance accuracy, rules as registered (3 classes) | 182 / 237 = 76.8% |
| Stance accuracy, first rules (before revision on dev) | 164 / 237 = 69.2% |
| Precision of the *rumour* stance | 100 / 117 = 85.5% [78%, 91%] |
| Recall of the *rumour* stance | 100 / 124 = 80.6% |
| Precision of the *announcement* stance | 62 / 76 = 81.6% [71%, 89%] |
| Linked player is right | 221 / 237 = 93.2% [89%, 96%] |
| A linked (player, club) claim names a club he is said to be going to | 221 / 269 = 82.2% [77%, 86%] |
| A claim the rules call a rumour is a rumour, with the right player and target club | 93 / 136 = 68.4% [60%, 76%] |

## What this means for the study

About two claims in three that enter the study are clean rumours about the right player and target
club. The rest are mostly the selling club or a rival named in the headline, interviews and
denials ("transfer itirafı", "açıklaması"), and departures. These claims are almost all labelled 0
(the player does not move to that club), so they add noise that makes every model's AUC lower, text
and metadata alike. The primary comparison is between models on the same claims, so the noise does
not favour either side, but absolute AUCs should be read with it in mind.

Main error types seen (for a later, separately registered revision; nothing is changed now):

- Interviews and statements about a transfer ("X'ten transfer açıklaması/itirafı", "menajerinden
  açıklama") are called rumours.
- Departures ("ayrılık", "görüşme odasına", "takımda kalacak mı") are called rumours.
- Agreements phrased as future or as the target's news ("istanbula geliyor", "imza atacağı tarih",
  "ile anlaştı" with another subject) are called rumours or other.
- "yeni transferi" on stories about an already signed player that are not about the move.
