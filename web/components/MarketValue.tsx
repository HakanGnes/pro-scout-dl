"use client";
import { useEffect, useMemo, useState } from "react";
import { fold, getJson } from "@/lib/data";

type M = { mae: number; median_pct: number; within_20: number; n: number };
type Row = [number, string, string, number, number, number, number, number, number, number];
type Data = { models: string[]; model_names: Record<string, string>; overall: Record<string, M>; by_anchor: Record<string, Record<string, number>>; rows: Row[]; players: number };

const eur = (v: number) => v >= 1e6 ? `€${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}m` : `€${Math.round(v / 1e3)}k`;
const anchorLabel = (a: string) => new Date(a + "T00:00:00").toLocaleDateString("en-GB", { month: "short", year: "numeric" });

export default function MarketValue() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [player, setPlayer] = useState<number | null>(null);

  useEffect(() => { getJson<Data>("market_value.json").then(setData).catch((e) => setError(String(e))); }, []);

  const people = useMemo(() => {
    const m = new Map<number, { name: string; key: string; top: number }>();
    for (const r of data?.rows ?? []) {
      const p = m.get(r[0]);
      if (!p) m.set(r[0], { name: r[1], key: fold(r[1]), top: r[3] });
      else p.top = Math.max(p.top, r[3]);
    }
    return m;
  }, [data]);
  const found = useMemo(() => {
    const q = fold(query);
    if (q.length < 2) return [];
    return [...people.entries()].filter(([, p]) => p.key.includes(q)).sort((a, b) => b[1].top - a[1].top).slice(0, 12);
  }, [query, people]);

  if (error) return <p className="error" role="alert">Could not load the predictions: {error}. Reload the page to try again.</p>;
  if (!data) return <p className="muted">Loading 31,220 predictions…</p>;
  const best = data.models.reduce((a, b) => (data.overall[a].mae <= data.overall[b].mae ? a : b));
  const rows = player === null ? [] : data.rows.filter((r) => r[0] === player);

  return (
    <div className="page market">
      <header className="page-head">
        <h1>Market value a year ahead</h1>
        <p>Given everything known about a player on a date, what will Transfermarkt say he is worth one year later? Five models
          made predictions for {data.rows.length.toLocaleString("en")} player-dates in 2023–2025, from 14 European first
          divisions, using only records from before each date.</p>
      </header>

      <section aria-label="Results">
        <table className="results">
          <thead><tr><th scope="col">Model</th><th scope="col">Typical error (log MAE)</th><th scope="col">Median error</th><th scope="col">Within ±20%</th></tr></thead>
          <tbody>
            {data.models.map((m) => (
              <tr key={m} className={m === best ? "best" : ""}>
                <th scope="row">{data.model_names[m]}</th>
                <td>{data.overall[m].mae.toFixed(3)}</td>
                <td>{Math.round(data.overall[m].median_pct * 100)}%</td>
                <td>{Math.round(data.overall[m].within_20 * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">Persistence: the value stays the same. Age curve: the median change for players of that age and position.
          The deep models beat LightGBM at every test date, but by little (1.6% of its error), and most of what can be predicted is
          already in the age curve.</p>
      </section>

      <section aria-label="Look up a player" className="lookup">
        <h2>Look up a player</h2>
        <label className="search"><span>Player name</span>
          <input type="search" value={query} onChange={(e) => { setQuery(e.target.value); setPlayer(null); }} placeholder="e.g. Bellingham, Kerem Aktürkoğlu" /></label>
        {player === null ? (
          query.trim().length >= 2 && (found.length ? (
            <ul className="options">
              {found.map(([id, p]) => (
                <li key={id}><button onClick={() => setPlayer(id)}><span className="opt-name">{p.name}</span>
                  <span className="opt-meta">up to {eur(p.top)}</span></button></li>
              ))}
            </ul>
          ) : <p className="empty">No player in the test years matches &ldquo;{query}&rdquo;. Only players in one of the 14 leagues in 2023–2025 are included.</p>)
        ) : (
          <>
            <h3>{people.get(player)?.name}</h3>
            <table className="player-preds">
              <thead><tr><th scope="col">Date</th><th scope="col">Value then</th><th scope="col">A year later</th>
                {data.models.map((m) => <th key={m} scope="col">{data.model_names[m]}</th>)}</tr></thead>
              <tbody>
                {rows.map((r) => {
                  const actual = r[4];
                  const preds = r.slice(5) as number[];
                  const closest = preds.reduce((bi, v, i) => (Math.abs(Math.log(v / actual)) < Math.abs(Math.log(preds[bi] / actual)) ? i : bi), 0);
                  return (
                    <tr key={r[2]}>
                      <th scope="row">{anchorLabel(r[2])}</th>
                      <td data-label="Value then">{eur(r[3])}</td>
                      <td data-label="A year later" className="actual">{eur(actual)}</td>
                      {preds.map((v, i) => <td key={i} data-label={data.model_names[data.models[i]]} className={i === closest ? "closest" : ""}>{eur(v)}</td>)}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="muted small">Underlined: the prediction closest to what Transfermarkt said a year later. Predictions are rounded to €1,000.</p>
            <button className="link" onClick={() => { setPlayer(null); setQuery(""); }}>Look up another player</button>
          </>
        )}
      </section>
    </div>
  );
}
