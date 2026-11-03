// The TypeScript networks and swaps must reproduce the Python results (test/golden.json, written by
// `python -m sim export-web --golden web/test/golden.json`) on the exported data.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadModels, predict } from "../lib/model";
import { loadProfiles, loadTeam, predictMatches, swapMany } from "../lib/sim";

const DATA = join(__dirname, "..", "public", "data");
const read = (f: string) => JSON.parse(readFileSync(join(DATA, f), "utf-8"));
const golden = JSON.parse(readFileSync(join(__dirname, "golden.json"), "utf-8"));
const models = loadModels(read("models.json"));
const profiles = loadProfiles(read("players.json"));

describe("parity with PyTorch", () => {
  it("every network predicts the same xG on the reference graphs", () => {
    const team = loadTeam(read(`teams/${golden.graphs[0].team_id}.json`));
    const graphs = golden.graphs.map((g: { match_id: number }) => team.matches.find((m) => m.matchId === g.match_id)!);
    const p = predict(models, graphs);
    golden.graphs.forEach((g: { pred: number[] }, j: number) => {
      g.pred.forEach((v, k) => expect(p[k][j]).toBeCloseTo(v, 5));
    });
  });

  it("swaps give the same change, matches and uncertainty", () => {
    for (const c of golden.swaps) {
      const team = loadTeam(read(`teams/${c.team_id}.json`));
      const before = predictMatches(models, team.matches);
      const swaps = new Map<number, number>(Object.entries(c.swaps).map(([k, v]) => [Number(k), v as number]));
      const r = swapMany(models, team, before, swaps, profiles);
      expect([r.matches, r.skipped, r.teamMatches]).toEqual([c.matches, c.skipped, c.team_matches]);
      expect(r.delta).toBeCloseTo(c.delta, 5);
      expect(r.seMatches).toBeCloseTo(c.se_matches, 5);
      expect(r.sdModels).toBeCloseTo(c.sd_models, 5);
      expect(r.before).toBeCloseTo(c.before, 5);
      expect(r.seasonDelta).toBeCloseTo(c.season_delta, 5);
      r.perMatch.forEach((m, i) => expect(m.delta).toBeCloseTo(c.per_match_delta[i], 5));
    }
  });
});
