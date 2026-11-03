// Runs the networks off the main thread so the board stays responsive while a swap is computed.
import { loadModels, Models, RawModels } from "./model";
import { loadProfiles, loadTeam, predictMatches, Profile, swapMany, SwapResult, Team } from "./sim";

export type Request =
  | { kind: "init"; base: string }
  | { kind: "team"; teamId: number }
  | { kind: "swap"; id: number; teamId: number; swaps: [number, number][] };

export type Response =
  | { kind: "ready"; players: number }
  | { kind: "team"; teamId: number; predicted: number; actual: number; matches: number }
  | { kind: "swap"; id: number; result: SwapResult; alone: (SwapResult | null)[] }
  | { kind: "error"; id?: number; message: string };

let models: Models | null = null;
let profiles: Map<number, Profile> | null = null;
let base = "";
const teams = new Map<number, { team: Team; before: number[][] }>();

async function getTeam(teamId: number) {
  let t = teams.get(teamId);
  if (!t) {
    const raw = await (await fetch(`${base}/teams/${teamId}.json`)).json();
    const team = loadTeam(raw);
    t = { team, before: predictMatches(models!, team.matches) };
    teams.set(teamId, t);
  }
  return t;
}

const post = (r: Response) => (self as unknown as Worker).postMessage(r);

self.onmessage = async (ev: MessageEvent<Request>) => {
  const msg = ev.data;
  try {
    if (msg.kind === "init") {
      base = msg.base;
      const [m, p] = await Promise.all([fetch(`${base}/models.json`).then((r) => r.json() as Promise<RawModels>),
        fetch(`${base}/players.json`).then((r) => r.json())]);
      models = loadModels(m);
      profiles = loadProfiles(p);
      post({ kind: "ready", players: profiles.size });
    } else if (msg.kind === "team") {
      const { team, before } = await getTeam(msg.teamId);
      const n = team.matches.length;
      const predicted = before.reduce((s, row) => s + row.reduce((a, v) => a + v, 0), 0) / (before.length * n);
      const actual = team.matches.reduce((s, m) => s + (m.target ?? 0), 0) / n;
      post({ kind: "team", teamId: msg.teamId, predicted, actual, matches: n });
    } else if (msg.kind === "swap") {
      const { team, before } = await getTeam(msg.teamId);
      const swaps = new Map(msg.swaps);
      const result = swapMany(models!, team, before, swaps, profiles!);
      const alone = msg.swaps.length > 1
        ? msg.swaps.map(([o, i]) => { try { return swapMany(models!, team, before, new Map([[o, i]]), profiles!); } catch { return null; } })
        : [];
      post({ kind: "swap", id: msg.id, result, alone });
    }
  } catch (e) {
    post({ kind: "error", id: "id" in msg ? msg.id : undefined, message: e instanceof Error ? e.message : String(e) });
  }
};
