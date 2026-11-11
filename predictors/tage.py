"""TAGE-style expert predictor, replayed over routing traces.

Design (chosen 2025-11-06):
  * base predictor      : the pairwise vote table from analysis/headroom.py (source experts at layer T-d
                          vote for target experts at layer T; ~32 KB on OLMoE).
  * tagged tables       : one per history length h in {1, 2, 4} layers. The key for target layer T at
                          horizon d is (T, token id, the top-m experts at each of layers T-d, ..., T-d-h+1).
                          m < k coarsens the history so long histories repeat. Each entry holds a tag, the
                          stored top-k set, a 2-bit confidence and a 2-bit useful counter (~8-9 bytes).
  * prediction          : longest matching table wins ("provider"); the next match or the base is the
                          alternate. CPU TAGE lets a confidence-0 (just allocated) provider defer to the
                          alternate; here that costs 3-4 points, so it is off unless --alt-on-na.
  * update              : provider conf +1 on a hit (recall >= hit_thresh) else -1; at conf 0 the stored set
                          is replaced by the true set. Useful bits track provider-vs-alternate; on a miss
                          (recall < alloc_thresh) allocate in the shortest longer table with u == 0, else
                          decay u of those tables. u is halved every reset_period updates.
  * prefetch-2k         : the k predicted experts plus the base table's next-best experts up to 2k.

Replays prefill + decode tokens in order (learning from all), scores decode tokens only, same as the
other predictors in analysis/headroom.py. Horizons: 1, 2, 4, 8 layers ahead.
"""
import argparse, json, glob, os, sys, time
from multiprocessing import get_context
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from headroom import load, true_topk, Pairwise, HORIZONS  # noqa: E402

MASK = (1 << 64) - 1


def mix(h, v):
    h ^= (v + 0x9E3779B97F4A7C15 + ((h << 6) & MASK) + (h >> 2)) & MASK
    h = (h * 0xBF58476D1CE4E5B9) & MASK
    return h ^ (h >> 31)


def key_hash(parts, salt):
    h = salt
    for p in parts:
        h = mix(h, int(p))
    return h


class TaggedTable:
    def __init__(self, n_entries, tag_bits, salt):
        self.n, self.tag_mask, self.salt = n_entries, (1 << tag_bits) - 1, salt
        self.tags = [-1] * n_entries
        self.sets = [()] * n_entries
        self.conf = [0] * n_entries
        self.u = [0] * n_entries

    def lookup(self, parts):
        h = key_hash(parts, self.salt)
        return h % self.n, (h >> 40) & self.tag_mask


class TAGE:
    def __init__(self, L, E, k, tables, size_kb, tag_bits=12, use_token=True,
                 hit_thresh=1.0, alloc_thresh=1.0, reset_period=65536, use_alt_on_na=False):
        self.L, self.E, self.k = L, E, k
        self.cfg = tables                      # list of (history_len_layers, top_m)
        self.base = Pairwise(L, E, k)
        set_bits = k * int(np.ceil(np.log2(E)))
        self.entry_bytes = (tag_bits + set_bits + 2 + 2) / 8
        n_per = max(16, int(size_kb * 1024 / len(tables) / self.entry_bytes))
        self.tables = [TaggedTable(n_per, tag_bits, salt=0x1234567 * (i + 1)) for i in range(len(tables))]
        self.use_token, self.hit_thresh, self.alloc_thresh = use_token, hit_thresh, alloc_thresh
        self.reset_period, self.use_alt_on_na = reset_period, use_alt_on_na
        self.n_updates = 0
        self.stats = {"base": [0, 0.0], **{f"T{i}(h={h},m={m})": [0, 0.0] for i, (h, m) in enumerate(tables)}}

    def kb(self):
        return sum(t.n for t in self.tables) * self.entry_bytes / 1024

    # ----- keys
    def keys(self, T, d, token_id, ranked):
        """ranked: [L, k] ranked expert ids for this token. Returns per-table (idx, tag) or None."""
        out = []
        for ti, (h, m) in enumerate(self.cfg):
            lo = T - d - h + 1
            if lo < 0:
                out.append(None); continue
            parts = [T, ti]
            if self.use_token:
                parts.append(token_id)
            for l in range(T - d, lo - 1, -1):
                parts.extend(sorted(ranked[l, :m].tolist()))
            out.append(self.tables[ti].lookup(parts))
        return out

    # ----- predict
    def predict(self, T, src_set, keys):
        base_scores = self.base.scores(T, src_set)
        base_pred = tuple(np.argpartition(-base_scores, self.k - 1)[: self.k].tolist())
        provider = alt = None
        for ti in range(len(self.tables) - 1, -1, -1):
            if keys[ti] is None:
                continue
            idx, tag = keys[ti]
            if self.tables[ti].tags[idx] == tag:
                if provider is None:
                    provider = ti
                elif alt is None:
                    alt = ti
                    break
        prov_pred = self.tables[provider].sets[keys[provider][0]] if provider is not None else base_pred
        alt_pred = self.tables[alt].sets[keys[alt][0]] if alt is not None else base_pred
        final = prov_pred
        if provider is not None and self.use_alt_on_na and self.tables[provider].conf[keys[provider][0]] == 0:
            final = alt_pred
        # prefetch-2k: final set + best remaining base experts
        extra = [e for e in np.argsort(-base_scores).tolist() if e not in final][: self.k]
        return dict(final=final, prov=prov_pred, alt=alt_pred, provider=provider, alt_i=alt,
                    base=base_pred, final2k=tuple(final) + tuple(extra))

    # ----- update
    def update(self, T, src_set, keys, true_set, p):
        k = self.k
        ts = set(true_set.tolist())
        r_final = len(ts.intersection(p["final"])) / k
        r_prov = len(ts.intersection(p["prov"])) / k
        r_alt = len(ts.intersection(p["alt"])) / k
        true_t = tuple(sorted(ts))
        prov = p["provider"]
        if prov is not None:
            tbl, idx = self.tables[prov], keys[prov][0]
            if r_prov >= self.hit_thresh:
                tbl.conf[idx] = min(tbl.conf[idx] + 1, 3)
            elif tbl.conf[idx] > 0:
                tbl.conf[idx] -= 1
            else:
                tbl.sets[idx] = true_t
            if set(p["prov"]) != set(p["alt"]):
                if r_prov > r_alt:
                    tbl.u[idx] = min(tbl.u[idx] + 1, 3)
                elif r_prov < r_alt:
                    tbl.u[idx] = max(tbl.u[idx] - 1, 0)
        if r_final < self.alloc_thresh:
            start = prov + 1 if prov is not None else 0
            cand = [ti for ti in range(start, len(self.tables)) if keys[ti] is not None]
            free = [ti for ti in cand if self.tables[ti].u[keys[ti][0]] == 0]
            if free:
                ti = free[0]
                tbl, (idx, tag) = self.tables[ti], keys[ti]
                tbl.tags[idx], tbl.sets[idx], tbl.conf[idx], tbl.u[idx] = tag, true_t, 0, 0
            else:
                for ti in cand:
                    idx = keys[ti][0]
                    self.tables[ti].u[idx] = max(self.tables[ti].u[idx] - 1, 0)
        self.n_updates += 1
        if self.n_updates % self.reset_period == 0:
            for tbl in self.tables:
                tbl.u = [u >> 1 for u in tbl.u]
        self.base.update(T, src_set, true_set)
        return r_final


# ------------------------------------------------------------------ replay
G = {}  # trace globals shared with forked workers


def replay(job):
    d, size_kb, cfg = job["d"], job["size_kb"], job["tables"]
    tk, rk, tok, seq, phase = G["tk"], G["rk"], G["token_id"], G["seq_id"], G["phase"]
    L, E, k = G["L"], G["E"], G["k"]
    N = len(tk)
    pr = TAGE(L, E, k, cfg, size_kb, tag_bits=job["tag_bits"], use_token=job["use_token"],
              hit_thresh=job["hit_thresh"], alloc_thresh=job["alloc_thresh"], use_alt_on_na=job.get("use_alt_on_na", False))
    h_t = h_t2 = h_b = 0.0; cnt = 0
    for n in range(N):
        dec = phase[n] == 1
        for T in range(d, L):
            src = tk[n, T - d]
            keys = pr.keys(T, d, tok[n], rk[n])
            p = pr.predict(T, src, keys)
            true = tk[n, T]
            if dec:
                ts = set(true.tolist())
                r = len(ts.intersection(p["final"])) / k
                h_t += r; h_t2 += len(ts.intersection(p["final2k"])) / k
                h_b += len(ts.intersection(p["base"])) / k; cnt += 1
                who = "base" if p["provider"] is None else list(pr.stats)[1 + p["provider"]]
                pr.stats[who][0] += 1; pr.stats[who][1] += r
            pr.update(T, src, keys, true, p)
    stats = {kk: dict(share=v[0] / max(cnt, 1), recall=v[1] / max(v[0], 1)) for kk, v in pr.stats.items()}
    return dict(d=d, size_kb=size_kb, tables=cfg, tage=h_t / cnt, tage_2k=h_t2 / cnt, pairwise=h_b / cnt,
                tagged_kb=pr.kb(), base_kb=pr.base.kb(), provider_stats=stats)


def ranked_topk(logits, k):
    return np.argsort(-logits.astype(np.float32), axis=-1)[..., :k]


def parse_tables(s):
    return [tuple(int(x) for x in t.split(":")) for t in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=sorted(glob.glob("traces/OLMoE*__*.npz")))
    ap.add_argument("--sizes-kb", nargs="+", type=int, default=[16, 64, 256])
    ap.add_argument("--tables", default="1:4,2:2,4:1", help="comma list of history_len:top_m, shortest first")
    ap.add_argument("--no-token", action="store_true", help="do not fold the token id into the keys")
    ap.add_argument("--tag-bits", type=int, default=12)
    ap.add_argument("--alt-on-na", action="store_true", help="CPU-TAGE rule: a just-allocated (conf 0) provider defers to the alternate. Off by default: measured 3-4 points worse here")
    ap.add_argument("--hit-thresh", type=float, default=0.75, help="provider recall >= this counts as a hit (1.0 = exact set)")
    ap.add_argument("--alloc-thresh", type=float, default=0.875, help="final recall < this triggers allocation")
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()

    tr, meta = load(args.traces)
    E, k, L = meta["num_experts"], meta["topk"], meta["num_moe_layers"]
    N = len(tr["token_id"]); dec = tr["phase"] == 1
    print(f"model={meta['model']} tokens={N} decode={dec.sum()} layers={L} experts={E} k={k}")
    G.update(tk=true_topk(tr["logits"], k), rk=ranked_topk(tr["logits"], k), token_id=tr["token_id"],
             seq_id=tr["seq_id"], phase=tr["phase"], L=L, E=E, k=k)

    # heuristic reference at each horizon (same as headroom.py)
    xl = tr["xlayer"]; heur = {}
    for d in args.horizons:
        r1, r2 = [], []
        for T in range(d, L):
            pred = xl[dec, T - d, d - 1, :]; true = G["tk"][dec, T]
            r1.append(np.mean([len(np.intersect1d(p[:k], t)) / k for p, t in zip(pred, true)]))
            r2.append(np.mean([len(np.intersect1d(p, t)) / k for p, t in zip(pred, true)]))
        heur[d] = dict(recall_k=float(np.mean(r1)), recall_2k=float(np.mean(r2)))

    cfg = parse_tables(args.tables)
    jobs = [dict(d=d, size_kb=s, tables=cfg, tag_bits=args.tag_bits, use_token=not args.no_token,
                 hit_thresh=args.hit_thresh, alloc_thresh=args.alloc_thresh, use_alt_on_na=args.alt_on_na)
            for d in args.horizons for s in args.sizes_kb]
    t0 = time.time()
    with get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(replay, jobs)
    print(f"replayed {len(jobs)} configs in {time.time()-t0:.0f}s  (tables={cfg}, token={'yes' if not args.no_token else 'no'}, alt_on_na={'yes' if args.alt_on_na else 'no'})")
    print(f"{'d':>3} {'size':>6} {'tagged':>7} | {'heur':>6} {'heur2k':>6} | {'pairw':>6} | {'TAGE':>6} {'TAGE2k':>6} | provider share / recall")
    for r in sorted(res, key=lambda r: (r["d"], r["size_kb"])):
        ps = "  ".join(f"{kk}:{v['share']:.2f}/{v['recall']:.2f}" for kk, v in r["provider_stats"].items())
        print(f"{r['d']:>3} {r['size_kb']:>5}K {r['tagged_kb']:>6.0f}K | {heur[r['d']]['recall_k']:.3f}  {heur[r['d']]['recall_2k']:.3f} | {r['pairwise']:.3f} | {r['tage']:.3f}  {r['tage_2k']:.3f} | {ps}")

    tag = meta["model"].split("/")[-1] + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(meta=meta, args=vars(args), heuristic=heur, results=res), open(f"{args.out}/tage_{tag}.json", "w"), indent=1, default=float)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.8))
    H = args.horizons
    ax.plot(H, [heur[d]["recall_k"] for d in H], "o-", label="next-layer router heuristic")
    ax.plot(H, [heur[d]["recall_2k"] for d in H], "o--", label="heuristic, prefetch 2k")
    big = max(args.sizes_kb)
    ax.plot(H, [next(r["pairwise"] for r in res if r["d"] == d and r["size_kb"] == big) for d in H], "s-", color="gray", label=f"pairwise base alone ({res[0]['base_kb']:.0f}KB)")
    for s in args.sizes_kb:
        ax.plot(H, [next(r["tage"] for r in res if r["d"] == d and r["size_kb"] == s) for d in H], "D-", label=f"TAGE, {s}KB tagged + base")
    ax.plot(H, [next(r["tage_2k"] for r in res if r["d"] == d and r["size_kb"] == big) for d in H], "D--", label=f"TAGE {big}KB, prefetch 2k")
    ax.set_xlabel("lookahead d (layers)"); ax.set_ylabel(f"recall@{k} on decode tokens"); ax.set_xticks(H); ax.set_ylim(0, 1)
    ax.set_title(f"{tag}: TAGE-style vs baselines (tables {args.tables})"); ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout(); plt.savefig(f"{args.out}/tage_{tag}.png", dpi=140)
    print(f"saved {args.out}/tage_{tag}.json and .png")


if __name__ == "__main__":
    main()
