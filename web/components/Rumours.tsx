"use client";
import { useEffect, useMemo, useState } from "react";
import { fold, getJson } from "@/lib/data";

type Band = { band: string; claims: number; predicted: number; happened: number };
type Pair = [string, string, string | null, number, number, string, string, number, number, number | null, string];
type Data = {
  pairs: Pair[];
  check_2026: { claims: number; claims_closed: number; closed_until: string; known: number; happened: number; auc_model: number; auc_model_95: [number, number]; reliability: Band[]; data_end: string; archive_end: string };
  test_2025: { claims: number; rate: number; auc: number; mean_p: number; reliability: Band[] };
};

const pct = (v: number) => `${Math.round(v * 100)}%`;
const day = (s: string) => new Date(s + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });

function Reliability({ title, rows, note }: { title: string; rows: Band[]; note: React.ReactNode }) {
  return (
    <figure className="reliability">
      <figcaption><h3>{title}</h3>{note}</figcaption>
      <table>
        <thead><tr><th scope="col">The model said</th><th scope="col">Claims</th><th scope="col">Predicted on average</th><th scope="col">Came true</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.band}>
              <th scope="row">{r.band}</th>
              <td>{r.claims.toLocaleString("en")}</td>
              <td><span className="bar predicted" style={{ width: `${r.predicted * 100}%` }} />{pct(r.predicted)}</td>
              <td><span className="bar happened" style={{ width: `${r.happened * 100}%` }} />{pct(r.happened)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

function outcomeOf(p: Pair): { cls: string; text: string } {
  if (p[9] === 1) return { cls: "happened", text: p[10].replace(/^happened \((.*)\)$/, "Happened: $1") };
  if (p[9] === 0) return { cls: "not", text: "Did not happen" };
  return { cls: "open", text: "Open: window not closed" };
}

export default function Rumours() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [show, setShow] = useState<"all" | "happened" | "not" | "open">("all");
  const [sort, setSort] = useState<"recent" | "probability" | "coverage">("recent");

  useEffect(() => { getJson<Data>("rumours.json").then(setData).catch((e) => setError(String(e))); }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    const q = fold(query);
    let r = data.pairs.filter((p) => !q || fold(p[0]).includes(q) || fold(p[1]).includes(q) || fold(p[2] ?? "").includes(q));
    if (show !== "all") r = r.filter((p) => outcomeOf(p).cls === show);
    const by = sort === "probability" ? (p: Pair) => -p[7] : sort === "coverage" ? (p: Pair) => -p[3] : (p: Pair) => -Date.parse(p[6]);
    return [...r].sort((a, b) => by(a) - by(b));
  }, [data, query, show, sort]);

  if (error) return <p className="error" role="alert">Could not load the rumours: {error}. Reload the page to try again.</p>;
  if (!data) return <p className="muted">Loading the rumours…</p>;
  const c = data.check_2026, t = data.test_2025;

  return (
    <div className="page rumours">
      <header className="page-head">
        <h1>Transfer rumours, 2026</h1>
        <p>For every rumour in eight Turkish sports papers that links a player to a club he is not at, the probability that
          Transfermarkt records the move within 180 days. The model does not read the wording: a registered test found it adds
          nothing measurable. It uses how many papers ran the pair in the last month, how long the pair has been in the news,
          and the player&rsquo;s value and age.</p>
      </header>

      <section className="checks" aria-label="How reliable the probabilities were">
        <Reliability title={`2025 test: AUC ${t.auc.toFixed(3)}`} rows={t.reliability}
          note={<p className="muted">{t.claims.toLocaleString("en")} rumours published in 2025, {pct(t.rate)} came true; the registered test.</p>} />
        <Reliability title={`2026 so far: AUC ${c.auc_model.toFixed(3)} (95% interval ${c.auc_model_95[0].toFixed(3)}–${c.auc_model_95[1].toFixed(3)})`} rows={c.reliability}
          note={<p className="muted">The {c.claims_closed.toLocaleString("en")} rumours up to {day(c.closed_until)}, whose 180 days have passed; {c.happened} came true.
            Transfermarkt&rsquo;s data ends on {day(c.data_end)}; after it, a move counts when at least 3 headlines from 2 papers report it done, so a move can be missed.</p>} />
      </section>

      <section aria-label="Rumoured moves">
        <h2>Every player–club pair, January to June 2026</h2>
        <p className="muted">Headlines up to {day(c.data_end)}, when the player data ends. Probability: the latest one, from the pair&rsquo;s most recent headline.
          A probability is not a verdict: of rumours given 20%, about one in five come true, and the model cannot say which.</p>
        <div className="table-tools">
          <label className="search"><span>Player or club</span>
            <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. Galatasaray, Osimhen" /></label>
          <label><span>Outcome</span>
            <select value={show} onChange={(e) => setShow(e.target.value as typeof show)}>
              <option value="all">All</option><option value="happened">Happened</option><option value="not">Did not happen</option><option value="open">Open</option>
            </select></label>
          <label><span>Order</span>
            <select value={sort} onChange={(e) => setSort(e.target.value as typeof sort)}>
              <option value="recent">Most recent first</option><option value="probability">Highest probability first</option><option value="coverage">Most headlines first</option>
            </select></label>
        </div>
        <p className="count">{rows.length.toLocaleString("en")} pairs</p>
        <table className="pairs">
          <thead><tr><th scope="col">Player</th><th scope="col">Rumoured club</th><th scope="col">Club then</th><th scope="col">Headlines</th>
            <th scope="col">Reported</th><th scope="col">Probability</th><th scope="col">Outcome</th></tr></thead>
          <tbody>
            {rows.map((p) => {
              const o = outcomeOf(p);
              return (
                <tr key={`${p[0]}-${p[1]}`}>
                  <th scope="row">{p[0]}</th>
                  <td data-label="Rumoured club">{p[1]}</td>
                  <td data-label="Club then">{p[2] ?? "–"}</td>
                  <td data-label="Headlines">{p[3]} in {p[4]} paper{p[4] === 1 ? "" : "s"}</td>
                  <td data-label="Reported">{p[5] === p[6] ? day(p[5]) : `${day(p[5])} – ${day(p[6])}`}</td>
                  <td data-label="Probability" className="prob"><span className="meter" style={{ width: `${p[7] * 100}%` }} />{pct(p[7])}</td>
                  <td data-label="Outcome" className={`outcome ${o.cls}`}>{o.text}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </div>
  );
}
