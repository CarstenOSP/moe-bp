"""Headroom study over routing traces: how predictable is expert routing, and how much
of that do caching and existing heuristics already capture?

All accuracy numbers are recall@k = |predicted set ∩ true top-k| / k, averaged over
DECODE tokens (prefill tokens still flow through caches/tables in order).
"""
import argparse, json, glob, os, hashlib
from collections import defaultdict, OrderedDict
import numpy as np

HORIZONS = [1, 2, 4, 8]


def load(paths):
    parts = []
    meta = None
    seq_off = 0
    for p in paths:
        z = np.load(p, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        meta = meta or m
        d = {k: z[k] for k in ["logits", "xlayer", "token_id", "seq_id", "pos", "phase"] + [k for k in ["xlayer_logits"] if k in z]}
        d["seq_id"] = d["seq_id"] + seq_off
        seq_off = d["seq_id"].max() + 1
        d["workload"] = np.array([m["workload"]] * len(d["token_id"]))
        parts.append(d)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return out, meta


def true_topk(logits, k):
    # [N, L, E] -> [N, L, k] sorted ids
    idx = np.argpartition(-logits.astype(np.float32), k, axis=-1)[..., :k]
    return np.sort(idx, axis=-1)


def recall(pred_sets, true_sets, k):
    """pred_sets, true_sets: [N, k'] / [N, k] int arrays -> mean |∩|/k."""
    hits = np.zeros(len(true_sets))
    for i in range(len(true_sets)):
        hits[i] = len(np.intersect1d(pred_sets[i][pred_sets[i] >= 0], true_sets[i])) / k
    return hits.mean()


# ---------------------------------------------------------------- online predictors
class CounterTable:
    """Branch-predictor-style table: key -> E saturating counters. Predict top-k counters.
    size_entries=None means unbounded (a ceiling for this history type)."""
    def __init__(self, E, k, size_entries=None, cmax=31):
        self.E, self.k, self.size, self.cmax = E, k, size_entries, cmax
        self.t = {}

    def _idx(self, key):
        if self.size is None:
            return key
        h = int(hashlib.blake2b(repr(key).encode(), digest_size=8).hexdigest(), 16)
        return h % self.size

    def predict(self, key, n=None):
        n = n or self.k
        c = self.t.get(self._idx(key))
        if c is None:
            return np.full(n, -1)
        return np.argpartition(-c, n - 1)[:n] if n < self.E else np.arange(self.E)

    def update(self, key, true_set):
        i = self._idx(key)
        c = self.t.get(i)
        if c is None:
            c = np.zeros(self.E, dtype=np.int16); self.t[i] = c
        c[true_set] += 1
        if c.max() >= self.cmax: c //= 2   # aging: halve the row when any counter saturates

    def bytes(self):
        per_entry = self.E // 2  # 4-bit counters
        return (self.size if self.size is not None else len(self.t)) * per_entry


def run_tables(tk, token_id, seq_id, phase, E, k, L, sizes_kb):
    """Replay online tables over the trace. Returns {name: {horizon: recall}}.
    Histories (what is known when predicting target layer T of token t):
      'xlayer-set'  : the expert set at layer T-d of the same token (d = horizon)
      'xlayer-set2' : expert sets at layers T-d and T-d-1
      'token'       : the token id of token t (known before layer 0 runs -> works for ALL layers, horizon=full)
      'prev-token'  : expert set at layer T of token t-1 (next-token horizon)
      'bimodal'     : per-layer global frequency
    """
    N = len(tk)
    res = defaultdict(dict)
    configs = []
    for d in HORIZONS:
        configs.append((f"xlayer-set d={d}", d, "set1", None))
    configs.append(("token-id (all layers)", 0, "token", None))
    for kb in sizes_kb:
        configs.append((f"token-id (all layers) {kb}KB", 0, "token", kb * 1024 // (E // 2)))
    configs.append(("prev-token-set (next token)", 0, "prev", None))
    configs.append(("bimodal (global freq)", 0, "bimodal", None))

    for name, d, kind, size in configs:
        tbl = CounterTable(E, k, size)
        hits, cnt = 0.0, 0
        prev_sets = {}  # layer -> set of previous token (same seq)
        last_seq = -1
        for n in range(N):
            if seq_id[n] != last_seq:
                prev_sets = {}; last_seq = seq_id[n]
            for T in range(L):
                if kind == "set1":
                    if T - d < 0: continue
                    key = (T, tuple(tk[n, T - d]))
                elif kind == "set2":
                    if T - d - 1 < 0: continue
                    key = (T, tuple(tk[n, T - d]), tuple(tk[n, T - d - 1]))
                elif kind == "token":
                    key = (T, int(token_id[n]))
                elif kind == "prev":
                    if T not in prev_sets: continue
                    key = (T, prev_sets[T])
                elif kind == "bimodal":
                    key = (T,)
                true = tk[n, T]
                if phase[n] == 1:
                    pred = tbl.predict(key)
                    hits += len(np.intersect1d(pred[pred >= 0], true)) / k; cnt += 1
                tbl.update(key, true)
            for T in range(L):
                prev_sets[T] = tuple(tk[n, T])
        res[name] = dict(recall=hits / max(cnt, 1), kb=tbl.bytes() / 1024, horizon=d)
        print(f"  {name:38s} recall@{k}={hits/max(cnt,1):.3f}  table={tbl.bytes()/1024:.0f}KB", flush=True)
    return res


class Pairwise:
    """Correlating predictor for sets: counters[T, e_src, e_dst]. Each expert in the source set
    votes for experts at the target layer; predict the top-k vote getters. Size = L*E*E 4-bit counters."""
    def __init__(self, L, E, k, cmax=31):
        self.c = np.zeros((L, E, E), dtype=np.int16); self.k, self.cmax, self.E = k, cmax, E

    def scores(self, T, src_set):
        return self.c[T, src_set].sum(0)

    def predict(self, T, src_set, n=None):
        s = self.scores(T, src_set); n = n or self.k
        return np.argpartition(-s, n - 1)[:n]

    def update(self, T, src_set, true):
        sub = self.c[T, src_set]
        sub[:, true] += 1
        if sub.max() >= self.cmax: sub //= 2
        self.c[T, src_set] = sub

    def kb(self): return self.c.size / 2 / 1024


def run_pairwise(tk, token_id, seq_id, phase, E, k, L, tok_kb=64):
    """Pairwise correlating predictor at each horizon (same token, layer T-d -> T), the next-token
    variant (layer T of token t-1 -> layer T of token t), and a tournament chooser between the
    pairwise predictor and a hashed token-id table (per layer, 2-bit chooser)."""
    N = len(tk); res = {}
    for d in HORIZONS + [0]:   # 0 = next-token (previous token's set at the same layer)
        pw = Pairwise(L, E, k)
        tok_tbl = CounterTable(E, k, tok_kb * 1024 // (E // 2))
        chooser = np.zeros(L, dtype=np.int8)   # >0 favors pairwise, <0 favors token table
        h_pw = h_tok = h_tour = h_pw2 = 0.0; cnt = 0
        prev = None; last_seq = -1
        for n in range(N):
            if seq_id[n] != last_seq: prev = None; last_seq = seq_id[n]
            for T in range(L):
                if d > 0:
                    if T - d < 0: continue
                    src = tk[n, T - d]
                else:
                    if prev is None: continue
                    src = prev[T]
                true = tk[n, T]; tkey = (T, int(token_id[n]))
                if phase[n] == 1:
                    p1 = pw.predict(T, src); p2 = tok_tbl.predict(tkey); p12 = pw.predict(T, src, 2 * k)
                    r1 = len(np.intersect1d(p1, true)) / k; r2 = len(np.intersect1d(p2[p2 >= 0], true)) / k
                    h_pw += r1; h_tok += r2; h_tour += r1 if chooser[T] >= 0 else r2; cnt += 1
                    h_pw2 += len(np.intersect1d(p12, true)) / k
                    chooser[T] = np.clip(chooser[T] + np.sign(r1 - r2), -3, 3)
                pw.update(T, src, true); tok_tbl.update(tkey, true)
            prev = tk[n]
        name = f"d={d}" if d > 0 else "next-token"
        res[name] = dict(pairwise=h_pw / max(cnt, 1), pairwise_2k=h_pw2 / max(cnt, 1), token=h_tok / max(cnt, 1), tournament=h_tour / max(cnt, 1), pairwise_kb=pw.kb(), token_kb=tok_kb)
        print(f"  {name:10s} pairwise({pw.kb():.0f}KB)={h_pw/max(cnt,1):.3f}  pairwise prefetch-2k={h_pw2/max(cnt,1):.3f}  token-id({tok_kb}KB)={h_tok/max(cnt,1):.3f}  tournament={h_tour/max(cnt,1):.3f}", flush=True)
    return res


def run_lru(tk, phase, L, k, sizes):
    """Per-layer LRU cache of expert ids. Returns {size: hit-rate over decode tokens}."""
    out = {}
    for C in sizes:
        caches = [OrderedDict() for _ in range(L)]
        hits, cnt = 0.0, 0
        for n in range(len(tk)):
            for T in range(L):
                c = caches[T]
                true = tk[n, T]
                if phase[n] == 1:
                    hits += sum(1 for e in true if e in c) / k; cnt += 1
                for e in true:
                    if e in c: c.move_to_end(e)
                    else:
                        c[e] = 1
                        if len(c) > C: c.popitem(last=False)
        out[C] = hits / max(cnt, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=sorted(glob.glob("traces/OLMoE*__*.npz")))
    ap.add_argument("--sizes-kb", nargs="+", type=int, default=[16, 64, 256])
    ap.add_argument("--out", default="results")
    ap.add_argument("--expert-mb", type=float, default=None, help="bytes per expert in MB for the cost model (default: OLMoE 3*2048*1024*2)")
    args = ap.parse_args()

    tr, meta = load(args.traces)
    E, k, L = meta["num_experts"], meta["topk"], meta["num_moe_layers"]
    N = len(tr["token_id"]); dec = tr["phase"] == 1
    print(f"model={meta['model']} tokens={N} decode={dec.sum()} layers={L} experts={E} k={k} workloads={sorted(set(tr['workload']))}")
    tk = true_topk(tr["logits"], k)  # [N, L, k]
    report = dict(meta=meta, n_tokens=int(N), n_decode=int(dec.sum()))

    # 1) usage skew per layer (decode tokens)
    print("\n[1] expert usage skew (decode tokens)")
    ent, cov16, cov32 = [], [], []
    for T in range(L):
        counts = np.bincount(tk[dec, T].ravel(), minlength=E).astype(float); p = counts / counts.sum()
        H = -(p[p > 0] * np.log2(p[p > 0])).sum() / np.log2(E)
        s = np.sort(p)[::-1]
        ent.append(H); cov16.append(s[:16].sum()); cov32.append(s[:32].sum())
    print(f"  normalized entropy per layer: min={min(ent):.2f} mean={np.mean(ent):.2f} max={max(ent):.2f}")
    print(f"  traffic covered by top-16 experts: mean={np.mean(cov16):.2f}; by top-32: mean={np.mean(cov32):.2f}")
    report["usage"] = dict(entropy=ent, cov16=cov16, cov32=cov32)

    # 2) LRU caching, previous-token reuse
    print("\n[2] caching baselines")
    lru = run_lru(tk, tr["phase"], L, k, [k, 16, 24, 32, 48])
    for C, h in lru.items(): print(f"  LRU cache of {C:2d} experts/layer: hit-rate={h:.3f}")
    reuse = []
    for n in range(1, N):
        if tr["seq_id"][n] == tr["seq_id"][n - 1] and dec[n]:
            reuse.append(np.mean([len(np.intersect1d(tk[n, T], tk[n - 1, T])) / k for T in range(L)]))
    print(f"  same experts as previous token (per layer, recall@{k}): {np.mean(reuse):.3f}")
    report["lru"] = lru; report["prev_token_reuse"] = float(np.mean(reuse))

    # 3) next-layer-router heuristic (Fate/HOBBIT-style) at every horizon
    print("\n[3] cross-layer router heuristic: layer T-d's input through layer T's router")
    xl = tr["xlayer"]  # [N, L, H, 2k]
    heur = {}
    for d in HORIZONS:
        r1, r2 = [], []
        for T in range(d, L):
            src = T - d
            pred = xl[dec, src, d - 1, :]              # [Ndec, 2k], first k are the top-k
            true = tk[dec, T]
            r1.append(recall(pred[:, :k], true, k)); r2.append(recall(pred, true, k))
        heur[d] = dict(recall_k=float(np.mean(r1)), recall_2k=float(np.mean(r2)), per_layer=[float(x) for x in r1])
        print(f"  d={d}: recall@{k}={np.mean(r1):.3f}  recall@{2*k} (prefetch 2k)={np.mean(r2):.3f}")
    report["heuristic"] = heur

    # 4) online branch-predictor-style tables
    print("\n[4] online counter tables (learn as they run, no training)")
    tables = run_tables(tk, tr["token_id"], tr["seq_id"], tr["phase"], E, k, L, args.sizes_kb)
    report["tables"] = tables
    print("\n[4b] pairwise correlating predictor (each source expert votes for target experts) + tournament")
    pair = run_pairwise(tk, tr["token_id"], tr["seq_id"], tr["phase"], E, k, L)
    report["pairwise"] = pair

    # 5) crude cost model: OLMoE expert = 3 * hidden * inter * 2 bytes; PCIe 25 GB/s
    expert_mb = args.expert_mb or 3 * 2048 * 1024 * 2 / 1e6
    ms_per_miss = expert_mb / 25e3 * 1e3  # MB / (MB/ms)
    def tok_ms(hit): return L * k * (1 - hit) * ms_per_miss
    cost = {"on-demand (no cache)": tok_ms(0.0), f"LRU {32}/layer": tok_ms(lru[32]),
            "heuristic d=1": tok_ms(heur[1]["recall_k"]), "heuristic d=4": tok_ms(heur[4]["recall_k"]),
            "table xlayer-set d=1": tok_ms(tables["xlayer-set d=1"]["recall"]),
            "table token-id": tok_ms(tables["token-id (all layers)"]["recall"]),
            "pairwise d=1": tok_ms(pair["d=1"]["pairwise"]), "pairwise d=4": tok_ms(pair["d=4"]["pairwise"]),
            "tournament d=1": tok_ms(pair["d=1"]["tournament"]), "tournament d=4": tok_ms(pair["d=4"]["tournament"]), "oracle": 0.0}
    print(f"\n[5] crude RAM->GPU fetch cost per decode token (expert={expert_mb:.1f}MB, PCIe 25GB/s, {ms_per_miss:.2f}ms/miss)")
    for name, ms in cost.items(): print(f"  {name:24s} {ms:6.1f} ms/token")
    report["cost_ms_per_token"] = cost

    os.makedirs(args.out, exist_ok=True)
    tag = meta["model"].split("/")[-1]
    json.dump(report, open(f"{args.out}/headroom_{tag}.json", "w"), indent=1, default=float)

    # ---- plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(HORIZONS, [heur[d]["recall_k"] for d in HORIZONS], "o-", label="next-layer router (Fate/HOBBIT-style)")
    ax[0].plot(HORIZONS, [heur[d]["recall_2k"] for d in HORIZONS], "o--", label="next-layer router, prefetch 2k")
    ax[0].plot(HORIZONS, [tables[f"xlayer-set d={d}"]["recall"] for d in HORIZONS], "s-", label="online table: expert set at layer T-d (unbounded)")
    ax[0].plot(HORIZONS, [pair[f"d={d}"]["pairwise"] for d in HORIZONS], "D-", label=f"online pairwise table ({pair['d=1']['pairwise_kb']:.0f}KB)")
    ax[0].plot(HORIZONS, [pair[f"d={d}"]["pairwise_2k"] for d in HORIZONS], "D--", label="online pairwise table, prefetch 2k")
    ax[0].plot(HORIZONS, [pair[f"d={d}"]["tournament"] for d in HORIZONS], "D:", label="tournament: pairwise vs token-id (64KB)")
    ax[0].axhline(tables["token-id (all layers)"]["recall"], color="k", ls="-.", label="token-id table (known before layer 0)")
    ax[0].axhline(tables["prev-token-set (next token)"]["recall"], color="gray", ls=":", label="prev-token set table (next-token horizon)")
    ax[0].axhline(lru[32], color="r", ls=":", label="LRU 32/layer")
    ax[0].set_xlabel("lookahead (layers)"); ax[0].set_ylabel(f"recall@{k} on decode tokens"); ax[0].set_xticks(HORIZONS); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"{tag}: accuracy vs horizon"); ax[0].legend(fontsize=7, loc="lower left")
    ax[1].plot(list(lru.keys()), list(lru.values()), "o-"); ax[1].set_xlabel("LRU cache size (experts/layer)"); ax[1].set_ylabel("hit rate"); ax[1].set_ylim(0, 1)
    ax[1].set_title("caching alone"); ax[1].axhline(1.0, color="k", lw=0.5)
    plt.tight_layout(); plt.savefig(f"{args.out}/headroom_{tag}.png", dpi=140)
    print(f"\nsaved {args.out}/headroom_{tag}.json and .png")


if __name__ == "__main__":
    main()
