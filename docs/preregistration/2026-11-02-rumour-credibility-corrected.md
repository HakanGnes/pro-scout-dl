# Pre-registered study, corrected run: do the words of a transfer rumour tell whether it comes true?

- **Registered:** 2026-11-02, before the corrected run. Committed with the code that runs it; that commit
  is the timestamp.
- **Status:** registered, not yet run. **Rule 4 switched off** (2026-11-02, before the run): 52 of the 80
  hand-labelled headlines it links name the right player (65%, 95% interval 54%–74%), below the 85% required.
  Correction to rule 4's text: the 80 headlines were fixed before labelling by the rule that linked them (field
  `rule` of the labelling items), but they are spread over items h401–h500, not h401–h480 as written. Counted by
  position, h401–h480 give 61 of 80 (76%); the rule is off either way. The 20 headlines linked by rule 2's
  "left the club" tie (kept when no other player is named): 16 of 20 right (80%).
- **Why a second run:** the first run ([`2026-10-29-rumour-credibility.md`](2026-10-29-rumour-credibility.md))
  linked a last-name-only player when he moved to or from a club named in the headline within a year before
  *or after* it. A later move could therefore decide whether a headline entered the study. Leaving those claims
  out of the first run's test predictions removes its primary result (+0.0095 [−0.0062, +0.0279]).
- **Disclosure — seen before this registration:** everything listed in the first registration; the first run's
  full test results, by model, paper and group; the sensitivity analysis above; for January–June 2026 rumours,
  the first model's AUC (0.75) and calibration against moves recorded by Transfermarkt or reported done in the
  archive; per linking rule, the share of claims that came true in all periods (the 2025 test period included).
  **The 2025 test period is therefore not untouched.** It is kept as the test period because no other period
  has known outcomes from Transfermarkt for a full 180 days; the result is a corrected re-analysis, not a fresh
  test, and is reported as such.

## What changes (everything else is as in the first registration)

**Linking** (`features/rumours/entities.py`, rules fixed at this commit):

1. Only records dated before the headline decide a link: a last-name-only player is tied to a named club if he
   plays there the day before, or moved to or from it in the year before. Values used to choose between
   namesakes are values up to that day.
2. A last-name-only player tied only by having *left* a named club is dropped when the headline names another
   player ("Fenerbahçe'den Vlahović hamlesi! Transferde Tadić detayı").
3. A full-name match needs the first name to be another word of the headline.
4. A last name with no tie is linked to the most valuable player active on the day (a market value in the two
   years before) who carries it, if he was worth at least €1m and at least five times any other active player
   with that name, is not known by one name only, and the word is not part of a club's name in the headline.
   **This rule is used only if** at least 85% of 80 hand-labelled headlines it links (2018–2024, drawn at random
   before labelling, items h401–h480 of the labelling sheet) name the right player. The share is computed and
   the switch `Gazetteer.UNIQUE_SURNAME` set accordingly in a commit before the run; both are reported.

**Archived pages:** publication dates are fetched for the rumour articles the corrected linking adds
(`python -m etl history-titles --urls data/processed/rumours/title_urls.txt`); the run requires 95% of them.

**Unchanged:** stance rules, label (180 days, 30-day grace), periods (train 2018–2022, validation July 2023 –
June 2024, test 2025), masking, metadata inputs, the six models and their settings, seeds, the primary outcome
AUC(fusion) − AUC(metadata) with success if the lower end of the pair-bootstrap 95% interval is above 0, and the
secondary outcomes. The input files have the same SHA-256 as in the first registration.

## Rules

- One run of `python -m models.rumours run` at the registering commit (plus the switch commit). Results replace
  `reports/rumours/README.md`, with the first run's results kept below them and marked as superseded.
- If the primary outcome fails, the downstream score uses the metadata model alone, as registered before.
- The model in the app is refitted with `python -m models.rumours fit` after the run.
