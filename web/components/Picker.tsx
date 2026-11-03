"use client";
// Choosing a replacement: the precomputed best candidates of his role, or anyone found by name.
import { useMemo, useState } from "react";
import { fold, Person, ROLE_NAME, Rankings, signed } from "@/lib/data";
import type { SquadPlayer } from "@/lib/sim";

type Props = {
  player: SquadPlayer; teamId: number; people: Map<number, Person>; rankings: Rankings | null;
  teamName: (id: number) => string; current: number | undefined;
  onChoose: (id: number) => void; onUndo: () => void; onClose: () => void;
};

export default function Picker({ player, teamId, people, rankings, teamName, current, onChoose, onUndo, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [sameRole, setSameRole] = useState(true);
  const [regular, setRegular] = useState(true);
  const best = rankings?.players[String(player.player_id)];
  const clubOf = (p: Person) => p.teamIds.map(teamName).filter(Boolean).join(", ");

  const found = useMemo(() => {
    const q = fold(query);
    if (q.length < 2) return [];
    return [...people.values()]
      .filter((p) => p.key.includes(q) && p.id !== player.player_id && !p.teamIds.includes(teamId))
      .filter((p) => !sameRole || p.role === player.role)
      .filter((p) => !regular || p.minutes >= 900)
      .sort((a, b) => b.minutes - a.minutes)
      .slice(0, 30);
  }, [query, people, player, teamId, sameRole, regular]);

  return (
    <div className="picker">
      <div className="picker-head">
        <div>
          <h2>Replace {player.name}</h2>
          <p>{ROLE_NAME[player.role ?? ""] ?? "Role unknown"}, {player.minutes.toLocaleString("en")} league minutes in {player.matches} matches</p>
        </div>
        <button className="close" onClick={onClose} aria-label="Close and go back to the result">Close</button>
      </div>
      {current !== undefined && (
        <p className="current">Now replaced by {people.get(current)?.name}. <button className="link" onClick={onUndo}>Put {player.name} back</button></p>
      )}

      <label className="search">
        <span>Find a player by name</span>
        <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. Kane, Griezmann, Özil" autoFocus />
      </label>
      <div className="filters">
        <label><input type="checkbox" checked={sameRole} onChange={(e) => setSameRole(e.target.checked)} /> Same role only</label>
        <label><input type="checkbox" checked={regular} onChange={(e) => setRegular(e.target.checked)} /> At least 900 league minutes</label>
      </div>

      {query.trim().length >= 2 ? (
        found.length ? (
          <ul className="options">
            {found.map((p) => (
              <li key={p.id}>
                <button onClick={() => onChoose(p.id)}>
                  <span className="opt-name">{p.name}</span>
                  <span className="opt-meta">{clubOf(p)}, {p.role ?? "?"}, {Math.round(p.minutes).toLocaleString("en")}&prime;</span>
                </button>
              </li>
            ))}
          </ul>
        ) : <p className="empty">No player from another team matches &ldquo;{query}&rdquo; with these filters. Clear a filter to see more.</p>
      ) : best ? (
        <>
          <h3>Best replacements from other teams</h3>
          <p className="note">All {best.candidates.toLocaleString("en")} {ROLE_NAME[best.role]?.toLowerCase()}s with at least {rankings!.min_minutes} league minutes, ranked by the change in npxG per match he played.</p>
          <ol className="options ranked">
            {best.best.map(([id, delta, se]) => {
              const p = people.get(id);
              return (
                <li key={id}>
                  <button onClick={() => onChoose(id)}>
                    <span className="opt-name">{p?.name ?? id}</span>
                    <span className="opt-meta">{p ? clubOf(p) : ""}</span>
                    <span className={`opt-delta ${delta >= 0 ? "gain" : "loss"}`} title={`± ${se.toFixed(3)} across matches`}>{signed(delta)}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </>
      ) : (
        <p className="empty">{rankings ? "No ranking for this player: he is not in the usual XI." : "Rankings for this team are not built yet."} Find a replacement by name above.</p>
      )}
    </div>
  );
}
