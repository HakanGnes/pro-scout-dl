// What the page shows (names, squads, rankings); the networks themselves run in lib/worker.ts.
import type { SquadPlayer } from "./sim";

export const DATA = "/data";

export type TeamIndex = { team_id: number; name: string; competition: string; matches: number };
export type Index = { teams: TeamIndex[]; n_models: number; models: string[]; source: string };
export type Person = { id: number; name: string; role: string | null; minutes: number; teamIds: number[]; key: string };
export type TeamView = { team_id: number; name: string; competition: string; squad: SquadPlayer[]; xi: number[]; links_per_match: [number, number, number][] };
export type Ranking = { role: string; candidates: number; best: [number, number, number, number, number][] };
export type Rankings = { min_minutes: number; players: Record<string, Ranking> };

/** Lower case without accents, so "ozil" finds "Özil" ("ı" has no decomposition, so it is mapped by hand). */
export function fold(s: string): string {
  return s.replace(/ı/g, "i").replace(/İ/g, "I").normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();
}

export async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${DATA}/${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json() as Promise<T>;
}

export async function getPeople(): Promise<Map<number, Person>> {
  const raw = await getJson<{ players: unknown[][] }>("players.json");
  const out = new Map<number, Person>();
  for (const r of raw.players) {
    const [id, name, role, minutes, , teamIds] = r as [number, string, string | null, number, number, number[]];
    out.set(id, { id, name, role, minutes, teamIds, key: fold(name) });
  }
  return out;
}

export async function getRankings(teamId: number): Promise<Rankings | null> {
  try {
    return await getJson<Rankings>(`rankings/${teamId}.json`);
  } catch {
    return null;
  }
}

export const ROLE_NAME: Record<string, string> = { GK: "Goalkeeper", DF: "Defender", MD: "Midfielder", FW: "Forward" };

export const signed = (v: number, digits = 3) => (Number.isFinite(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}` : "–");
