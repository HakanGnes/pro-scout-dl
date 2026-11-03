// Player swaps, as in sim/swap.py (Simulator.swap_many): each swapped player on the pitch gets his
// replacement's season profile scaled to his minutes; passing links and starter flag stay with the slot.

import { decodeF32, decodeI32, Graph, Models, normalise, predictOne } from "./model";

export const NODE = {
  role: { GK: 0, DF: 1, MD: 2, FW: 3 } as Record<string, number>,
  minutes_share: 4, is_starter: 5, log_passes: 6, log_received: 7, pass_accuracy: 8,
  median_x: 9, median_y: 10, log_final_third: 11, log_box: 12, log_progressive: 13, log_touches: 14,
};
export const N_FEATURES = 15;
const COUNT = ["log_passes", "log_received", "log_final_third", "log_box", "log_progressive", "log_touches"] as const;

export type Profile = {
  id: number; name: string; role: string | null; minutes: number; matches: number; teamIds: number[];
  rates: Record<(typeof COUNT)[number], number>; passAccuracy: number; medianX: number; medianY: number;
};

export type Match = Graph & { matchId: number; gameweek: number; side: string; target: number | null; playerIds: number[] };

export type SquadPlayer = { player_id: number; name: string; role: string | null; matches: number; starts: number; minutes: number; x: number; y: number };

export type Team = {
  teamId: number; name: string; competition: string; squad: SquadPlayer[]; xi: number[];
  linksPerMatch: [number, number, number][]; matches: Match[];
};

type RawPlayers = { fields: string[]; players: unknown[][] };

export function loadProfiles(raw: RawPlayers): Map<number, Profile> {
  const out = new Map<number, Profile>();
  for (const r of raw.players) {
    const [id, name, role, minutes, matches, teamIds, ...rest] = r as [number, string, string | null, number, number, number[], ...number[]];
    const rates = Object.fromEntries(COUNT.map((c, i) => [c, rest[i]])) as Profile["rates"];
    out.set(id, { id, name, role, minutes, matches, teamIds, rates, passAccuracy: rest[6], medianX: rest[7], medianY: rest[8] });
  }
  return out;
}

type RawMatch = { match_id: number; gameweek: number; side: string; target: number | null; player_ids: number[]; x: string; edge_index: string; edge_attr: string; n_nodes: number; n_edges: number };
type RawTeam = { team_id: number; name: string; competition: string; squad: SquadPlayer[]; xi: number[]; links_per_match: [number, number, number][]; matches: RawMatch[] };

export function loadTeam(raw: RawTeam): Team {
  return {
    teamId: raw.team_id, name: raw.name, competition: raw.competition, squad: raw.squad, xi: raw.xi,
    linksPerMatch: raw.links_per_match,
    matches: raw.matches.map((m) => ({
      matchId: m.match_id, gameweek: m.gameweek, side: m.side, target: m.target, playerIds: m.player_ids,
      nNodes: m.n_nodes, nEdges: m.n_edges, x: decodeF32(m.x), edgeIndex: decodeI32(m.edge_index), edgeAttr: decodeF32(m.edge_attr),
    })),
  };
}

/** The profile's node vector for a slot of the given length (sim/profiles.py Profile.node_features). */
export function nodeFeatures(p: Profile, minutesShare: number, isStarter: boolean): Float32Array {
  const x = new Float32Array(N_FEATURES);
  if (p.role !== null && p.role in NODE.role) x[NODE.role[p.role]] = 1;
  x[NODE.minutes_share] = minutesShare;
  x[NODE.is_starter] = isStarter ? 1 : 0;
  for (const c of COUNT) x[NODE[c]] = Math.log1p(p.rates[c] * minutesShare);
  x[NODE.pass_accuracy] = p.passAccuracy;
  x[NODE.median_x] = p.medianX;
  x[NODE.median_y] = p.medianY;
  return x;
}

const share = (m: Match, k: number) => m.x[k * N_FEATURES + NODE.minutes_share];
const onPitch = (m: Match, pid: number) => {
  const k = m.playerIds.indexOf(pid);
  return k >= 0 && share(m, k) > 0;
};

export function applySwaps(m: Match, swaps: Map<number, Profile>): Match {
  const x = new Float32Array(m.x);
  const ids = [...m.playerIds];
  for (const [pid, prof] of swaps) {
    const k = m.playerIds.indexOf(pid);
    if (k < 0 || share(m, k) <= 0) continue;
    x.set(nodeFeatures(prof, share(m, k), m.x[k * N_FEATURES + NODE.is_starter] > 0.5), k * N_FEATURES);
    ids[k] = prof.id;
  }
  return { ...m, x, playerIds: ids };
}

export type SwapResult = {
  matches: number; skipped: number; teamMatches: number;
  delta: number; seMatches: number; sdModels: number; before: number; actual: number; seasonDelta: number;
  perNet: number[];            // each network's mean change per affected match
  perMatch: { matchId: number; gameweek: number; swapped: number; before: number; after: number; delta: number; actual: number | null }[];
};

const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const sd = (a: number[]) => {
  if (a.length < 2) return NaN;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
};

/** Predictions of every network on each match, before any swap; cached by the caller per team. */
export function predictMatches(models: Models, matches: Match[]): number[][] {
  const inputs = matches.map((m) => normalise(models, m));
  return models.nets.map((net) => matches.map((m, j) => predictOne(net, inputs[j].x, inputs[j].e, m)));
}

export function swapMany(models: Models, team: Team, before: number[][], swaps: Map<number, number>, profiles: Map<number, Profile>): SwapResult {
  if (swaps.size === 0) throw new Error("no swaps given");
  if (new Set(swaps.values()).size < swaps.size) throw new Error("the same player cannot come in twice");
  const prof = new Map<number, Profile>();
  for (const [out, cand] of swaps) {
    const p = profiles.get(cand);
    if (!p) throw new Error(`no profile for player ${cand}`);
    prof.set(out, p);
  }
  const games = team.matches.map((m, j) => ({ m, j })).filter(({ m }) => [...swaps.keys()].some((p) => onPitch(m, p)));
  const keep = games.filter(({ m }) => ![...swaps].some(([out, c]) =>
    m.playerIds.includes(c) && !(swaps.has(c) && onPitch(m, c)) && c !== out));
  const swapped = keep.map(({ m }) => applySwaps(m, prof));
  const after = predictMatches(models, swapped);             // [net][kept match]
  const nNets = models.nets.length;
  const delta = after.map((row, k) => row.map((v, i) => v - before[k][keep[i].j]));
  const perMatch = keep.map(({ m, j }, i) => ({
    matchId: m.matchId, gameweek: m.gameweek, swapped: [...swaps.keys()].filter((p) => onPitch(m, p)).length,
    before: mean(before.map((row) => row[j])), after: mean(after.map((row) => row[i])),
    delta: mean(delta.map((row) => row[i])), actual: m.target,
  }));
  const n = keep.length;
  const perNet = delta.map((row) => mean(row));
  return {
    matches: n, skipped: games.length - n, teamMatches: team.matches.length,
    delta: n ? mean(perNet) : NaN,
    seMatches: n > 1 ? sd(perMatch.map((r) => r.delta)) / Math.sqrt(n) : NaN,
    sdModels: n && nNets > 1 ? sd(perNet) : NaN,
    before: n ? mean(perMatch.map((r) => r.before)) : NaN,
    actual: n ? mean(perMatch.map((r) => r.actual ?? NaN)) : NaN,
    seasonDelta: team.matches.length ? perMatch.reduce((s, r) => s + r.delta, 0) / team.matches.length : NaN,
    perNet: n ? perNet : [],
    perMatch,
  };
}
