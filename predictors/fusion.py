"""Fusion of expert predictors: rank experts by a weighted sum of several sources' per-expert scores,
with the source weights learned online per layer (multiplicative weights, no training).

Sources (each gives a score per expert in [0, 1] for target layer T at horizon d):
  heur : softmax over the hidden-state heuristic's top-2k logits (layer T-d input through layer T's
         router, stored in the trace as xlayer / xlayer_logits); 0 outside the 2k candidates.
  base : pairwise vote table (experts at T-d vote for experts at T), votes / sum.
  tage : TAGE tagged-entry stored set when a tag matches, scaled by (conf + 1) / 4; absent otherwise.

Fused score = sum_i w[T, i] * p_i. Prefetch set = top-k (or top-2k) fused. After the true set is
known, each present source's own top-k recall r_i is computed and w[T, i] *= exp(eta * (r_i - mean r)),
renormalized. Also reports each source alone, an oracle-selection bound (best single source per
prediction), and a confidence-threshold sweep (fetch all experts with fused score >= tau * max score,
at least k) as recall vs. average experts fetched.
"""
import argparse, json, glob, os, sys, time
from multiprocessing import get_context
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from headroom import load, true_topk, HORIZONS  # noqa: E402
from tage import TAGE, ranked_topk, parse_tables  # noqa: E402

G = {}
TAUS = [0.9, 0.7, 0.5, 0.3, 0.15]


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def replay(job):
    d, sources, eta = job["d"], job["sources"], job["eta"]
    tk, rk, tok, phase = G["tk"], G["rk"], G["token_id"], G["phase"]
    xl, xv = G["xlayer"], G["xlayer_logits"]
    L, E, k = G["L"], G["E"], G["k"]
    N = len(tk)
    S = len(sources)
    use_tage = "tage" in sources
    pr = TAGE(L, E, k, job["tables"], job["tage_kb"], use_token=True, hit_thresh=0.75, alloc_thresh=0.875, use_alt_on_na=False)
    w = np.full((L, S), 1.0 / S)
    hits = {"fused": 0.0, "fused_2k": 0.0, "oracle": 0.0, **{s: 0.0 for s in sources}}
    tau_hits = {t: 0.0 for t in TAUS}; tau_cnt = {t: 0.0 for t in TAUS}
    cnt = 0; present = np.zeros(S)
    for n in range(N):
        dec = phase[n] == 1
        for T in range(d, L):
            src = tk[n, T - d]
            keys = pr.keys(T, d, tok[n], rk[n])
            p = pr.predict(T, src, keys)
            true = tk[n, T]; ts = set(true.tolist())
            # ---- per-source score vectors
            P = np.zeros((S, E)); avail = np.ones(S, dtype=bool)
            for i, s in enumerate(sources):
                if s == "heur":
                    ids = xl[n, T - d, d - 1, :]; lg = xv[n, T - d, d - 1, :].astype(np.float32)
                    P[i, ids] = softmax(lg)
                elif s == "base":
                    v = pr.base.scores(T, src).astype(np.float64)
                    P[i] = v / v.sum() if v.sum() > 0 else 0.0
                elif s == "tage":
                    if p["provider"] is None:
                        avail[i] = False
                    else:
                        conf = pr.tables[p["provider"]].conf[keys[p["provider"]][0]]
                        P[i, list(p["prov"])] = (conf + 1) / 4.0
            fused = (w[T][:, None] * P).sum(0)
            order = np.argsort(-fused)
            pred_k, pred_2k = order[:k], order[: 2 * k]
            # ---- score
            r_src = np.array([len(ts.intersection(np.argpartition(-P[i], k - 1)[:k].tolist())) / k if avail[i] else np.nan for i in range(S)])
            if dec:
                hits["fused"] += len(ts.intersection(pred_k.tolist())) / k
                hits["fused_2k"] += len(ts.intersection(pred_2k.tolist())) / k
                hits["oracle"] += np.nanmax(r_src)
                for i, s in enumerate(sources):
                    if avail[i]:
                        hits[s] += r_src[i]; present[i] += 1
                mx = fused.max()
                for t in TAUS:
                    sel = order[: max(k, int((fused >= t * mx).sum()))]
                    tau_hits[t] += len(ts.intersection(sel.tolist())) / k; tau_cnt[t] += len(sel)
                cnt += 1
            # ---- learn weights (sources present only)
            m = avail & ~np.isnan(r_src)
            if m.sum() >= 2:
                r = r_src[m]; upd = np.exp(eta * (r - r.mean()))
                w[T, m] *= upd; w[T] /= w[T].sum()
            pr.update(T, src, keys, true, p)
    res = dict(d=d, sources=sources, eta=eta, fused=hits["fused"] / cnt, fused_2k=hits["fused_2k"] / cnt,
               oracle_select=hits["oracle"] / cnt,
               single={s: hits[s] / max(present[i], 1) for i, s in enumerate(sources)},
               coverage={s: present[i] / cnt for i, s in enumerate(sources)},
               weights_mean={s: float(w[d:, i].mean()) for i, s in enumerate(sources)},
               tau_curve={t: dict(recall=tau_hits[t] / cnt, avg_fetched=tau_cnt[t] / cnt) for t in TAUS})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=sorted(glob.glob("traces/OLMoE*__*.npz")))
    ap.add_argument("--combos", nargs="+", default=["heur,base", "heur,base,tage", "base,tage"])
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--tables", default="1:4,2:2,4:1")
    ap.add_argument("--tage-kb", type=int, default=64)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tr, meta = load(args.traces)
    assert "xlayer_logits" in tr, "traces lack xlayer_logits; re-collect with the current collect_traces.py"
    E, k, L = meta["num_experts"], meta["topk"], meta["num_moe_layers"]
    N = len(tr["token_id"]); dec = tr["phase"] == 1
    print(f"model={meta['model']} tokens={N} decode={dec.sum()} layers={L} experts={E} k={k}")
    G.update(tk=true_topk(tr["logits"], k), rk=ranked_topk(tr["logits"], k), token_id=tr["token_id"],
             phase=tr["phase"], xlayer=tr["xlayer"], xlayer_logits=tr["xlayer_logits"], L=L, E=E, k=k)
    cfg = parse_tables(args.tables)
    jobs = [dict(d=d, sources=c.split(","), eta=args.eta, tables=cfg, tage_kb=args.tage_kb) for c in args.combos for d in args.horizons]
    t0 = time.time()
    with get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(replay, jobs)
    print(f"replayed {len(jobs)} configs in {time.time()-t0:.0f}s (eta={args.eta}, tage {args.tage_kb}KB tables {cfg})")
    print(f"{'sources':16s} {'d':>2} | {'fused':>6} {'fus2k':>6} | {'oracle':>6} | singles (recall / coverage) | mean weights")
    for r in sorted(res, key=lambda r: (args.combos.index(",".join(r["sources"])), r["d"])):
        singles = "  ".join(f"{s}:{r['single'][s]:.3f}/{r['coverage'][s]:.2f}" for s in r["sources"])
        wts = " ".join(f"{s}:{r['weights_mean'][s]:.2f}" for s in r["sources"])
        print(f"{','.join(r['sources']):16s} {r['d']:>2} | {r['fused']:.3f}  {r['fused_2k']:.3f} | {r['oracle_select']:.3f} | {singles} | {wts}")
    print("\nconfidence-threshold sweep (sources=heur,base,tage): recall @ avg experts fetched per layer")
    for r in sorted(res, key=lambda r: r["d"]):
        if r["sources"] == ["heur", "base", "tage"]:
            print(f"  d={r['d']}: " + "  ".join(f"tau={t}: {v['recall']:.3f}@{v['avg_fetched']:.1f}" for t, v in r["tau_curve"].items()))

    tag = meta["model"].split("/")[-1] + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(meta=meta, args=vars(args), results=res), open(f"{args.out}/fusion_{tag}.json", "w"), indent=1, default=float)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    H = args.horizons
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    full = [r for r in res if r["sources"] == ["heur", "base", "tage"]]
    get = lambda key, rs: [next(x[key] for x in rs if x["d"] == d) for d in H]
    ax[0].plot(H, [next(x["single"]["heur"] for x in full if x["d"] == d) for d in H], "o-", label="heuristic alone")
    ax[0].plot(H, [next(x["single"]["base"] for x in full if x["d"] == d) for d in H], "s-", color="gray", label="pairwise base alone")
    for c, style in [("heur,base", "^-"), ("heur,base,tage", "D-"), ("base,tage", "v-")]:
        rs = [r for r in res if ",".join(r["sources"]) == c]
        if rs: ax[0].plot(H, get("fused", rs), style, label=f"fused: {c}")
    ax[0].plot(H, get("oracle_select", full), "k:", label="oracle select (best single source)")
    ax[0].plot(H, get("fused_2k", full), "D--", label="fused heur,base,tage, prefetch 2k")
    ax[0].set_xlabel("lookahead d (layers)"); ax[0].set_ylabel(f"recall@{k} on decode tokens"); ax[0].set_xticks(H); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"{tag}: online fusion"); ax[0].legend(fontsize=7, loc="lower left")
    for r in full:
        xs = [v["avg_fetched"] for v in r["tau_curve"].values()]; ys = [v["recall"] for v in r["tau_curve"].values()]
        ax[1].plot(xs, ys, "o-", label=f"d={r['d']}")
    ax[1].axvline(k, color="gray", ls=":"); ax[1].axvline(2 * k, color="gray", ls=":")
    ax[1].set_xlabel("avg experts fetched per layer"); ax[1].set_ylabel(f"recall@{k}"); ax[1].set_ylim(0, 1)
    ax[1].set_title("confidence-threshold prefetch (fused)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{args.out}/fusion_{tag}.png", dpi=140)
    print(f"saved {args.out}/fusion_{tag}.json and .png")


if __name__ == "__main__":
    main()
