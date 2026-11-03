"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getJson, getPeople, getRankings, Index, Person, Rankings, TeamView } from "@/lib/data";
import type { SwapResult } from "@/lib/sim";
import type { Request, Response } from "@/lib/worker";
import Pitch from "./Pitch";
import Picker from "./Picker";
import Result from "./Result";

// competition names as the Wyscout dataset has them
const LEAGUES = ["English first division", "Spanish first division", "Italian first division", "German first division", "French first division"];
const LEAGUE_NAME: Record<string, string> = {
  "English first division": "Premier League", "Spanish first division": "La Liga", "Italian first division": "Serie A",
  "German first division": "Bundesliga", "French first division": "Ligue 1",
};

function useVertical() {
  const [v, setV] = useState(false);
  useEffect(() => {
    const q = window.matchMedia("(max-width: 720px)");
    const on = () => setV(q.matches);
    on();
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, []);
  return v;
}

export type Baseline = { predicted: number; actual: number; matches: number };

export default function Board() {
  const worker = useRef<Worker | null>(null);
  const reqId = useRef(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [index, setIndex] = useState<Index | null>(null);
  const [people, setPeople] = useState<Map<number, Person> | null>(null);
  const [league, setLeague] = useState(LEAGUES[0]);
  const [teamId, setTeamId] = useState<number | null>(null);
  const [team, setTeam] = useState<TeamView | null>(null);
  const [baseline, setBaseline] = useState<Baseline | null>(null);
  const [rankings, setRankings] = useState<Rankings | null>(null);
  const [swaps, setSwaps] = useState<Map<number, number>>(new Map());
  const [selected, setSelected] = useState<number | null>(null);
  const [result, setResult] = useState<{ total: SwapResult; alone: (SwapResult | null)[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [showLinks, setShowLinks] = useState(true);
  const vertical = useVertical();
  const slip = useRef<HTMLElement | null>(null);

  // on a phone the slip is below the board: bring the picker into view when a player is picked
  useEffect(() => {
    if (selected !== null && vertical) slip.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selected, vertical]);

  useEffect(() => {
    const w = new Worker(new URL("../lib/worker.ts", import.meta.url));
    worker.current = w;
    w.onmessage = (ev: MessageEvent<Response>) => {
      const m = ev.data;
      if (m.kind === "ready") setReady(true);
      else if (m.kind === "team") setBaseline({ predicted: m.predicted, actual: m.actual, matches: m.matches });
      else if (m.kind === "swap") {
        if (m.id === reqId.current) { setResult({ total: m.result, alone: m.alone }); setBusy(false); }
      } else if (m.kind === "error") { setError(m.message); setBusy(false); }
    };
    w.postMessage({ kind: "init", base: `${window.location.origin}/data` } satisfies Request);
    Promise.all([getJson<Index>("index.json"), getPeople()]).then(([i, p]) => {
      setIndex(i);
      setPeople(p);
      const arsenal = i.teams.find((t) => t.name === "Arsenal") ?? i.teams[0];
      setLeague(arsenal.competition);
      setTeamId(arsenal.team_id);
    }).catch((e) => setError(String(e)));
    return () => w.terminate();
  }, []);

  useEffect(() => {
    if (teamId === null) return;
    setTeam(null); setBaseline(null); setRankings(null); setSwaps(new Map()); setSelected(null); setResult(null);
    getJson<TeamView>(`teams/${teamId}.json`).then(setTeam).catch((e) => setError(String(e)));
    getRankings(teamId).then(setRankings);
  }, [teamId]);

  useEffect(() => {
    if (ready && teamId !== null) worker.current?.postMessage({ kind: "team", teamId } satisfies Request);
  }, [ready, teamId]);

  useEffect(() => {
    if (!ready || teamId === null) return;
    if (swaps.size === 0) { setResult(null); setBusy(false); return; }
    const id = ++reqId.current;
    setBusy(true);
    worker.current?.postMessage({ kind: "swap", id, teamId, swaps: [...swaps] } satisfies Request);
  }, [swaps, ready, teamId]);

  const names = useCallback((id: number) => people?.get(id)?.name ?? `player ${id}`, [people]);
  const teamName = useCallback((id: number) => index?.teams.find((t) => t.team_id === id)?.name ?? "", [index]);
  const xi = useMemo(() => team ? team.xi.map((id) => team.squad.find((p) => p.player_id === id)!).filter(Boolean) : [], [team]);
  const bench = useMemo(() => team ? team.squad.filter((p) => !team.xi.includes(p.player_id) && p.minutes >= 90) : [], [team]);
  const leagueTeams = useMemo(() => index?.teams.filter((t) => t.competition === league) ?? [], [index, league]);
  const selectedPlayer = team?.squad.find((p) => p.player_id === selected) ?? null;

  const choose = (incoming: number) => {
    if (selected === null) return;
    const next = new Map(swaps);
    next.set(selected, incoming);
    setSwaps(next);
    setSelected(null);
  };
  const undo = (out: number) => {
    const next = new Map(swaps);
    next.delete(out);
    setSwaps(next);
  };

  return (
    <div className="board-page">
      <header className="masthead">
        <div className="title">
          <h1>Tactics board</h1>
          <p>Replace players in a 2017/18 squad and see how ten passing-network models expect the team&rsquo;s chances to change.</p>
        </div>
        <div className="pickers">
          <div className="leagues" role="tablist" aria-label="League">
            {LEAGUES.map((l) => (
              <button key={l} role="tab" aria-selected={league === l} className={league === l ? "on" : ""}
                      onClick={() => { setLeague(l); const t = index?.teams.find((x) => x.competition === l); if (t) setTeamId(t.team_id); }}>
                {LEAGUE_NAME[l]}
              </button>
            ))}
          </div>
          <label className="team-select">
            <span>Team</span>
            <select value={teamId ?? ""} onChange={(e) => setTeamId(Number(e.target.value))}>
              {leagueTeams.map((t) => <option key={t.team_id} value={t.team_id}>{t.name}</option>)}
            </select>
          </label>
        </div>
      </header>

      {error && <p className="error" role="alert">Could not load the board data: {error}. Reload the page to try again.</p>}

      <div className="workspace">
        <section className="surface" aria-label="Board">
          {team ? (
            <>
              <Pitch xi={xi} links={team.links_per_match} swaps={swaps} names={names} selected={selected}
                     onSelect={(id) => setSelected(selected === id ? null : id)} vertical={vertical} showLinks={showLinks} />
              <div className="under-pitch">
                <label className="toggle">
                  <input type="checkbox" checked={showLinks} onChange={(e) => setShowLinks(e.target.checked)} />
                  Passing links (line width: completed passes per match, both ways)
                </label>
                <p className="hint">Magnets sit at each player&rsquo;s median position over the season; the team attacks {vertical ? "upwards" : "to the right"}.</p>
              </div>
            </>
          ) : <div className="loading">Loading the squad…</div>}
        </section>

        {team && bench.length > 0 && (
          <section className="bench" aria-label="Rest of the squad">
          <h2>Rest of the squad</h2>
          <ul>
            {bench.map((p) => {
              const incoming = swaps.get(p.player_id);
              return (
                <li key={p.player_id}>
                  <button className={["chip", incoming !== undefined && "swapped", selected === p.player_id && "selected"].filter(Boolean).join(" ")}
                          aria-pressed={selected === p.player_id}
                          onClick={() => setSelected(selected === p.player_id ? null : p.player_id)}>
                    <span className="chip-name">{incoming !== undefined ? names(incoming) : p.name}</span>
                    <span className="chip-meta">{p.role ?? "?"} {p.minutes.toLocaleString("en")}&prime;</span>
                  </button>
                </li>
              );
            })}
          </ul>
          </section>
        )}

        <aside className="slip" ref={slip} aria-live="polite">
          {selectedPlayer && people ? (
            <Picker player={selectedPlayer} teamId={teamId!} people={people} rankings={rankings} teamName={teamName}
                    current={swaps.get(selectedPlayer.player_id)} onChoose={choose}
                    onUndo={() => { undo(selectedPlayer.player_id); setSelected(null); }} onClose={() => setSelected(null)} />
          ) : (
            <Result team={team} baseline={baseline} swaps={swaps} names={names} result={result} busy={busy || (swaps.size > 0 && !ready)}
                    nModels={index?.n_models ?? 10} onUndo={undo} onReset={() => setSwaps(new Map())} />
          )}
        </aside>
      </div>
    </div>
  );
}
