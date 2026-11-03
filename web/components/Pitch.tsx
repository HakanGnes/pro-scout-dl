"use client";
// The board: a pitch drawn like the printed lines of a magnetic tactics board, with one magnet per player of the
// team's usual XI at his season median position, and the season's passing links between them.
import { useMemo } from "react";
import type { SquadPlayer } from "@/lib/sim";

const L = 105, W = 68;   // metres

type Props = {
  xi: SquadPlayer[];
  links: [number, number, number][];
  swaps: Map<number, number>;
  names: (id: number) => string;
  selected: number | null;
  onSelect: (id: number) => void;
  vertical: boolean;
  showLinks: boolean;
};

export const shortName = (name: string) => {
  const parts = name.split(" ");
  return parts.length > 1 && /\.$/.test(parts[0]) ? parts.slice(1).join(" ") : name;
};

/** Season median positions overlap (two centre-backs, two holding midfielders); push magnets apart a little. */
function spread(points: { id: number; x: number; y: number }[], min = 11.5) {
  const p = points.map((q) => ({ ...q }));
  for (let it = 0; it < 60; it++) {
    for (let i = 0; i < p.length; i++)
      for (let j = i + 1; j < p.length; j++) {
        const dx = p[j].x - p[i].x, dy = p[j].y - p[i].y;
        const d = Math.hypot(dx, dy) || 0.01;
        if (d < min) {
          const push = (min - d) / 2, ux = dx / d, uy = dy / d;
          p[i].x -= ux * push; p[i].y -= uy * push; p[j].x += ux * push; p[j].y += uy * push;
        }
      }
    for (const q of p) { q.x = Math.min(L - 4, Math.max(4, q.x)); q.y = Math.min(W - 4, Math.max(4, q.y)); }
  }
  return new Map(p.map((q) => [q.id, q]));
}

export default function Pitch({ xi, links, swaps, names, selected, onSelect, vertical, showLinks }: Props) {
  const pos = useMemo(() => spread(xi.map((p) => ({ id: p.player_id, x: p.x * L, y: p.y * W }))), [xi]);
  // attack to the right; on a phone the pitch stands up and the team attacks upwards
  const at = (x: number, y: number) => (vertical ? { cx: y, cy: L - x } : { cx: x, cy: y });
  const box = vertical ? `-2 -2 ${W + 4} ${L + 4}` : `-2 -2 ${L + 4} ${W + 4}`;
  const maxPass = Math.max(1, ...links.map((l) => l[2]));
  const line = (x1: number, y1: number, x2: number, y2: number, k: string) => {
    const a = at(x1, y1), b = at(x2, y2);
    return <line key={k} x1={a.cx} y1={a.cy} x2={b.cx} y2={b.cy} />;
  };
  const rect = (x: number, y: number, w: number, h: number, k: string) => {
    const a = at(x, y), b = at(x + w, y + h);
    return <rect key={k} x={Math.min(a.cx, b.cx)} y={Math.min(a.cy, b.cy)} width={Math.abs(b.cx - a.cx)} height={Math.abs(b.cy - a.cy)} />;
  };
  const centre = at(L / 2, W / 2);

  return (
    <svg className="pitch" viewBox={box} role="group" aria-label="Team on the board">
      <g className="markings">
        {rect(0, 0, L, W, "outline")}
        {line(L / 2, 0, L / 2, W, "half")}
        <circle cx={centre.cx} cy={centre.cy} r={9.15} />
        <circle cx={centre.cx} cy={centre.cy} r={0.5} className="spot" />
        {rect(0, (W - 40.32) / 2, 16.5, 40.32, "box-l")}
        {rect(L - 16.5, (W - 40.32) / 2, 16.5, 40.32, "box-r")}
        {rect(0, (W - 18.32) / 2, 5.5, 18.32, "six-l")}
        {rect(L - 5.5, (W - 18.32) / 2, 5.5, 18.32, "six-r")}
        {rect(-1.6, (W - 7.32) / 2, 1.6, 7.32, "goal-l")}
        {rect(L, (W - 7.32) / 2, 1.6, 7.32, "goal-r")}
      </g>
      <g className="direction" aria-hidden="true">
        {(() => {
          const a = at(L / 2 + 14, W + 1.2), b = at(L / 2 + 24, W + 1.2);
          return <line x1={a.cx} y1={a.cy} x2={b.cx} y2={b.cy} markerEnd="url(#arrow)" />;
        })()}
      </g>
      <defs>
        <marker id="arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto">
          <path d="M0,0 L6,3 L0,6 z" className="arrowhead" />
        </marker>
      </defs>
      {showLinks && (
        <g className="links" aria-hidden="true">
          {links.map(([a, b, v]) => {
            const p = pos.get(a), q = pos.get(b);
            if (!p || !q) return null;
            const s = at(p.x, p.y), t = at(q.x, q.y);
            return <line key={`${a}-${b}`} x1={s.cx} y1={s.cy} x2={t.cx} y2={t.cy} strokeWidth={0.25 + 1.6 * (v / maxPass)} />;
          })}
        </g>
      )}
      <g className="magnets">
        {xi.map((p) => {
          const q = pos.get(p.player_id)!;
          const c = at(q.x, q.y);
          const incoming = swaps.get(p.player_id);
          const label = shortName(incoming !== undefined ? names(incoming) : p.name);
          const cls = ["magnet", incoming !== undefined && "swapped", selected === p.player_id && "selected"].filter(Boolean).join(" ");
          return (
            <g key={p.player_id} className={cls} transform={`translate(${c.cx} ${c.cy})`} role="button" tabIndex={0}
               aria-label={incoming !== undefined ? `${p.name}, replaced by ${names(incoming)}` : `Replace ${p.name}`}
               aria-pressed={selected === p.player_id}
               onClick={() => onSelect(p.player_id)}
               onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(p.player_id); } }}>
              <circle r={3.4} className="ring" />
              <circle r={2.7} className="disc" />
              <text y={6.6} className="name">{label}</text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
