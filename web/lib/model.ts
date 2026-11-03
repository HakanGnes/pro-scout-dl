// The passing-network models (DeepSets and GINE GNN) in plain TypeScript, inference only.
// Mirrors models/gnn/net.py: same layers, same order; weights come from `python -m sim export-web`.
// Dropout is off at inference, so it is left out.

export type Matrix = { rows: number; cols: number; data: Float32Array };

export type Graph = {
  nNodes: number;
  nEdges: number;
  x: Float32Array;          // nNodes × 15, raw node features
  edgeIndex: Int32Array;    // 2 × nEdges, row 0 = source, row 1 = target
  edgeAttr: Float32Array;   // nEdges × 2, raw edge features
};

type Linear = { w: Matrix; b: Float32Array };
type Mlp = [Linear, Linear];
type Conv = { eps: number; nn: Mlp; lin: Linear };
export type Net = { name: string; arch: "deepsets" | "gnn"; encoder: Mlp; head: Mlp; convs: Conv[]; norms: { w: Float32Array; b: Float32Array }[] };
export type Models = { nets: Net[]; xMean: Float32Array; xStd: Float32Array; eMean: Float32Array; eStd: Float32Array };

export function decodeF32(b64: string): Float32Array {
  const bytes = decodeBytes(b64);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

export function decodeI32(b64: string): Int32Array {
  const bytes = decodeBytes(b64);
  return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function decodeBytes(b64: string): Uint8Array {
  const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  const out = new Uint8Array(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

type RawParam = { shape: number[]; data: string };
type RawNet = { name: string; arch: "deepsets" | "gnn"; params: Record<string, RawParam> };
export type RawModels = { x_mean: string; x_std: string; e_mean: string; e_std: string; layers: number; nets: RawNet[] };

export function loadModels(raw: RawModels): Models {
  const nets = raw.nets.map((n) => {
    const p = (k: string) => {
      const v = n.params[k];
      if (!v) throw new Error(`${n.name}: missing ${k}`);
      return v;
    };
    const lin = (prefix: string): Linear => {
      const w = p(`${prefix}.weight`);
      return { w: { rows: w.shape[0], cols: w.shape[1], data: decodeF32(w.data) }, b: decodeF32(p(`${prefix}.bias`).data) };
    };
    const mlp = (prefix: string): Mlp => [lin(`${prefix}.0`), lin(`${prefix}.3`)];
    const convs: Conv[] = [];
    const norms: { w: Float32Array; b: Float32Array }[] = [];
    if (n.arch === "gnn") {
      for (let i = 0; i < raw.layers; i++) {
        convs.push({ eps: decodeF32(p(`convs.${i}.eps`).data)[0], nn: mlp(`convs.${i}.nn`), lin: lin(`convs.${i}.lin`) });
        norms.push({ w: decodeF32(p(`norms.${i}.weight`).data), b: decodeF32(p(`norms.${i}.bias`).data) });
      }
    }
    return { name: n.name, arch: n.arch, encoder: mlp("encoder"), head: mlp("head"), convs, norms };
  });
  return { nets, xMean: decodeF32(raw.x_mean), xStd: decodeF32(raw.x_std), eMean: decodeF32(raw.e_mean), eStd: decodeF32(raw.e_std) };
}

// y = x Wᵀ + b for n rows of x (n × in) -> n × out
function linear(x: Float64Array, n: number, l: Linear): Float64Array {
  const { rows: out, cols: inp, data: w } = l.w;
  const y = new Float64Array(n * out);
  for (let r = 0; r < n; r++) {
    const xo = r * inp, yo = r * out;
    for (let o = 0; o < out; o++) {
      let s = l.b[o];
      const wo = o * inp;
      for (let i = 0; i < inp; i++) s += x[xo + i] * w[wo + i];
      y[yo + o] = s;
    }
  }
  return y;
}

function relu(a: Float64Array): Float64Array {
  for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0;
  return a;
}

function mlp(x: Float64Array, n: number, m: Mlp): Float64Array {
  return linear(relu(linear(x, n, m[0])), n, m[1]);
}

function layerNorm(x: Float64Array, n: number, d: number, w: Float32Array, b: Float32Array): Float64Array {
  const y = new Float64Array(n * d);
  for (let r = 0; r < n; r++) {
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[r * d + i];
    mean /= d;
    let v = 0;
    for (let i = 0; i < d; i++) v += (x[r * d + i] - mean) ** 2;
    const inv = 1 / Math.sqrt(v / d + 1e-5);
    for (let i = 0; i < d; i++) y[r * d + i] = (x[r * d + i] - mean) * inv * w[i] + b[i];
  }
  return y;
}

const softplus = (v: number) => (v > 20 ? v : Math.log1p(Math.exp(v)));

export function normalise(models: Models, g: Graph): { x: Float64Array; e: Float64Array } {
  const nf = models.xMean.length, ef = models.eMean.length;
  const x = new Float64Array(g.nNodes * nf);
  for (let i = 0; i < x.length; i++) x[i] = (g.x[i] - models.xMean[i % nf]) / models.xStd[i % nf];
  const e = new Float64Array(g.nEdges * ef);
  for (let i = 0; i < e.length; i++) e[i] = (g.edgeAttr[i] - models.eMean[i % ef]) / models.eStd[i % ef];
  return { x, e };
}

/** One network's predicted non-penalty xG for one team graph. */
export function predictOne(net: Net, x: Float64Array, e: Float64Array, g: Graph): number {
  const n = g.nNodes;
  let h = relu(mlp(x, n, net.encoder));
  const d = h.length / n;
  for (let c = 0; c < net.convs.length; c++) {
    const conv = net.convs[c];
    const el = linear(e, g.nEdges, conv.lin);            // edge features mapped to hidden size
    const agg = new Float64Array(n * d);
    for (let k = 0; k < g.nEdges; k++) {
      const src = g.edgeIndex[k], dst = g.edgeIndex[g.nEdges + k];
      for (let i = 0; i < d; i++) {
        const m = h[src * d + i] + el[k * d + i];
        if (m > 0) agg[dst * d + i] += m;
      }
    }
    for (let i = 0; i < agg.length; i++) agg[i] += (1 + conv.eps) * h[i];
    const upd = relu(layerNorm(mlp(agg, n, conv.nn), n, d, net.norms[c].w, net.norms[c].b));
    const next = new Float64Array(n * d);
    for (let i = 0; i < next.length; i++) next[i] = h[i] + upd[i];
    h = next;
  }
  const pooled = new Float64Array(2 * d);
  for (let r = 0; r < n; r++) for (let i = 0; i < d; i++) pooled[i] += h[r * d + i];
  for (let i = 0; i < d; i++) {
    pooled[d + i] = pooled[i] / n;          // mean pool
    pooled[i] = pooled[i] / 11;             // sum pool / 11
  }
  return softplus(mlp(pooled, 1, net.head)[0]);
}

/** Every network's prediction for each graph: result[net][graph]. */
export function predict(models: Models, graphs: Graph[]): number[][] {
  const inputs = graphs.map((g) => normalise(models, g));
  return models.nets.map((net) => graphs.map((g, j) => predictOne(net, inputs[j].x, inputs[j].e, g)));
}
