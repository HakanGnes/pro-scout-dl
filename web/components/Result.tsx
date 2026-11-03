"use client";
// The result slip: what the swaps change, and how sure the models are.
import { signed, TeamView } from "@/lib/data";
import type { SwapResult } from "@/lib/sim";
import type { Baseline } from "./Board";

type Props = {
  team: TeamView | null; baseline: Baseline | null; swaps: Map<number, number>; names: (id: number) => string;
  result: { total: SwapResult; alone: (SwapResult | null)[] } | null; busy: boolean; nModels: number;
  onUndo: (out: number) => void; onReset: () => void;
};

function NetStrip({ values, nModels }: { values: number[]; nModels: number }) {
  // each network's estimate on one scale, with zero marked
  const lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  const pad = (hi - lo) * 0.12 || 0.01;
  const x = (v: number) => 4 + ((v - lo + pad) / (hi - lo + 2 * pad)) * 192;
  return (
    <figure className="netstrip">
      <svg viewBox="0 0 200 34" role="img" aria-label={`Estimates of the ${nModels} networks, from ${signed(Math.min(...values))} to ${signed(Math.max(...values))}`}>
        <line x1={4} x2={196} y1={17} y2={17} className="axis" />
        <line x1={x(0)} x2={x(0)} y1={6} y2={28} className="zero" />
        <text x={x(0)} y={4.5} className="zero-label">0</text>
        {values.map((v, i) => (
          i < nModels / 2
            ? <circle key={i} cx={x(v)} cy={17} r={3} className="net deepsets" />
            : <rect key={i} x={x(v) - 2.7} y={14.3} width={5.4} height={5.4} className="net gnn" />
        ))}
      </svg>
      <figcaption>Each mark is one network: circles DeepSets, squares GNN (five seeds each).</figcaption>
    </figure>
  );
}

export default function Result({ team, baseline, swaps, names, result, busy, nModels, onUndo, onReset }: Props) {
  if (!team) return <div className="result"><p className="muted">Loading…</p></div>;
  const r = result?.total;
  const list = [...swaps];

  return (
    <div className="result">
      {list.length === 0 ? (
        <>
          <h2>{team.name}, 2017/18</h2>
          {baseline ? (
            <dl className="figures">
              <div><dt>Predicted npxG per match</dt><dd>{baseline.predicted.toFixed(2)}</dd></div>
              <div><dt>Actual npxG per match</dt><dd>{baseline.actual.toFixed(2)}</dd></div>
              <div><dt>League matches</dt><dd>{baseline.matches}</dd></div>
            </dl>
          ) : <p className="muted">Running the models on the season…</p>}
          <p className="lead">Pick a magnet, or a player from the rest of the squad, to replace him.</p>
          <p className="muted">You can replace several players; the models then see all the changes in each match together.</p>
        </>
      ) : (
        <>
          <ul className="swaps">
            {list.map(([out, inn], i) => {
              const alone = result?.alone[i];
              return (
                <li key={out}>
                  <div>
                    <strong>{names(inn)}</strong>
                    <span className="for">for {names(out)}</span>
                  </div>
                  {alone && <span className={`alone ${alone.delta >= 0 ? "gain" : "loss"}`} title="This swap alone, per match he played">{signed(alone.delta)}</span>}
                  <button className="link" onClick={() => onUndo(out)} aria-label={`Undo: put ${names(out)} back`}>Undo</button>
                </li>
              );
            })}
          </ul>
          {list.length > 1 && <p className="muted small">Numbers beside each swap: that swap alone.</p>}

          {r && !busy ? (
            r.matches > 0 ? (
              <>
                <p className="change-label">Change in non-penalty xG per match {list.length > 1 ? "in matches with a replaced player" : "he played"}</p>
                <p className={`change ${r.delta >= 0 ? "gain" : "loss"}`}>{signed(r.delta)}</p>
                <p className="spread">
                  ± {r.seMatches.toFixed(3)} across matches, ± {r.sdModels.toFixed(3)} across the {nModels} networks
                </p>
                <NetStrip values={r.perNet} nModels={nModels} />
                <dl className="figures">
                  <div><dt>Predicted before</dt><dd>{r.before.toFixed(2)}</dd></div>
                  <div><dt>Predicted after</dt><dd>{(r.before + r.delta).toFixed(2)}</dd></div>
                  <div><dt>Actual</dt><dd>{r.actual.toFixed(2)}</dd></div>
                  <div><dt>Over the whole season</dt><dd>{signed(r.seasonDelta)}</dd></div>
                </dl>
                <p className="muted small">
                  {r.matches} of {r.teamMatches} league matches{r.skipped > 0 ? `; ${r.skipped} left out because the incoming player was already in the team that day` : ""}.
                  The season figure counts matches without a replaced player as no change.
                </p>
              </>
            ) : (
              <p className="empty">No match can be simulated: in every match the incoming player was already in the team. Choose someone else.</p>
            )
          ) : <p className="muted computing">Running {nModels} networks on every match…</p>}

          <button className="reset" onClick={onReset}>Clear all swaps</button>
        </>
      )}

      <div className="caveat">
        <h3>How to read this</h3>
        <p>A replacement brings his season profile from his own team, scaled to the minutes of the player he replaces: passes, touches, how often he gets the ball into the final third and the box. Passing links stay with the position.</p>
        <p>Shots are not among the inputs, so the models value creating chances, not finishing them: a striker who passes little is undervalued. This is a simulation under the models&rsquo; assumptions, not a forecast.</p>
      </div>
    </div>
  );
}
