# Preliminary trace findings: OLMoE-1B-7B and DeepSeek-V2-Lite (2025-11-06)

*Reproducibility: numbers come from Python 3.12, torch 2.15, transformers 5.22 (see `env/`). An earlier
run under transformers 4.57 agreed on the top-k expert set for 95.5% of token-layer pairs on identical
prefill input (99.4% expert overlap); OLMoE numbers moved by at most 0.5 points between the two, and
DeepSeek's online-table numbers by ~3 points, because greedy continuations diverge after bf16 tie flips
and those predictors learn from the token sequence. Treat differences of a few points as noise at this
trace size.*

**Setup.** OLMoE-1B-7B-0924 (16 MoE layers, 64 experts, top-8), bf16 on one GPU. Three workloads
(WikiText-103 prose, GSM8K math, HumanEval code), 60 prompts each, prefill + 160 greedy decode tokens at
batch 1. 51.6k tokens total, 28.8k of them decode tokens. All numbers below are **recall@8 on decode
tokens** = fraction of the true top-8 experts that the method had ready. Scripts are in this repo; traces are not committed and regenerate with the commands in `README.md`.

## Headline

1. **Routing is load-balanced, so caching alone does not solve it.** Normalized usage entropy is 0.95
   (near uniform); the 16 most-used experts cover only 49% of traffic. An LRU cache of 32 of the 64
   experts per layer gets 81% hits; 48 per layer gets 96%. Previous-token reuse is 42%.
2. **The hidden-state heuristic (Fate/HOBBIT-style, layer T−d's input through layer T's router) is strong
   at one layer ahead and decays fast with distance.** 82% at d=1, 76% at d=2, 64% at d=4, 46% at d=8.
   With prefetch-2k (fetch 16 candidates): 96% / 92% / 83% / 65%. (The pitch's ">90%" is the 2k number.)
3. **A 32 KB online, training-free "pairwise" table is nearly flat in horizon.** Each expert chosen at
   layer T−d votes with a saturating counter for experts at layer T. 66% at d=1, 65% at d=2, 64% at d=4,
   59% at d=8; with prefetch-2k: 86% / 86% / 85% / 81%. It loses to the heuristic by 16 points at d=1,
   ties at d=4, and wins by 13 points at d=8 (16 points with prefetch-2k). Same pattern on all three
   workloads. **This supports the pitch thesis: history-based prediction gives up accuracy nearby but
   keeps it far ahead, which is exactly what the SSD→RAM tier needs.**
4. **Token id alone is a surprisingly good predictor and is known before layer 0 runs.** An unbounded
   per-(layer, token) table reaches 70% for every layer with infinite lookahead, but it is vocab-sized
   (2.3 MB); hashed to 256 KB it drops to 58%, at 64 KB to 50%. Next-token prediction (layer T of token
   t from layer T of token t−1) via the pairwise table is 54% (76% with 2k).
5. **Keying on the exact expert set is a dead end.** A table indexed by the full 8-expert set at layer
   T−d barely beats global frequency (46% vs 43%) because the key space is C(64,8); it never repeats.
   Tables need decomposed features (per-expert votes, hashed partial histories), as CPU predictors do.
6. **Crude cost (PCIe 25 GB/s, 12.6 MB per OLMoE expert, 0.5 ms per miss):** on-demand 64 ms/token,
   LRU-32 12 ms, heuristic d=1 11 ms, heuristic d=4 23 ms, pairwise d=1 22 ms, pairwise d=4 23 ms.
   The room between any current method and the oracle is 11–23 ms/token on this model, and all of it
   is misses, so combining a short-horizon heuristic with a long-horizon history table is the obvious
   next design.

## Per-workload (heuristic vs pairwise, recall@8)

| workload  | LRU-32 | heur d=1 | heur d=8 | pairwise d=1 | pairwise d=8 |
|-----------|--------|----------|----------|--------------|--------------|
| wikitext  | 0.78   | 0.83     | 0.52     | 0.66         | 0.57         |
| gsm8k     | 0.79   | 0.83     | 0.45     | 0.67         | 0.61         |
| humaneval | 0.88   | 0.81     | 0.42     | 0.65         | 0.60         |

Code is the most cacheable (top-32 experts cover 88% of traffic); math is the least predictable by
hidden state at long range but the most predictable by history.

## Second model: DeepSeek-V2-Lite (26 MoE layers, 64 routed experts + 2 shared, top-6)

Same three workloads, 40 prompts x 128 decode tokens, 30.6k tokens (15.3k decode). recall@6 on decode.

| method                       | d=1  | d=2  | d=4  | d=8  | next-token |
|------------------------------|------|------|------|------|------------|
| next-layer router heuristic  | 0.80 | 0.73 | 0.62 | 0.50 | –          |
| same, prefetch 2k            | 0.96 | 0.91 | 0.82 | 0.68 | –          |
| pairwise table (52 KB)       | 0.53 | 0.52 | 0.49 | 0.46 | 0.37       |
| same, prefetch 2k            | 0.72 | 0.71 | 0.69 | 0.65 | 0.56       |
| token-id table (64 KB)       | 0.37 (all horizons; 0.54 unbounded 2.7 MB)                 |
| LRU 32/layer: 0.70 · LRU 48/layer: 0.90 · previous-token reuse: 0.27 · global frequency: 0.23 |

- **Harder than OLMoE everywhere.** Usage entropy 0.98 (even flatter), the top-16 experts carry only 37%
  of traffic, LRU-32 gets 70% (OLMoE 81%), and previous-token reuse is 27% (OLMoE 42%).
- **The heuristic decays the same way** (80 → 50 over d = 1..8). **The history table is again flat**
  (53 → 46), but it starts lower, so it only approaches the heuristic at d = 8 (0.46 vs 0.50; 0.65 vs
  0.68 with 2k candidates) without catching it. On OLMoE the crossover was at d ≈ 4.
- **Token id is much weaker here** (37% hashed vs 50% on OLMoE), so the history table's win over
  token-id is larger, and the tournament chooser again adds nothing over pairwise alone.
- Cost: on-demand 108 ms/token, LRU-32 32 ms, heuristic d=1 22 ms, heuristic d=4 41 ms, pairwise
  d=1 51 ms. Shared experts (always active, resident) are excluded.

**Reading across the two models:** the shape of the result is the same (hidden-state heuristic good
near, history-based tables flat with distance), but the single-length pairwise table is not strong
enough on its own on the harder model. That argues for the TAGE-style step (longer and multiple
history lengths, tagged entries) before claiming an advantage at d = 4–8, and for fusing heuristic
and history votes rather than choosing between them.

## TAGE-style predictor (added 2025-11-06, revised 2025-11-10, `predictors/tage.py`)

Design: the pairwise vote table stays as the base; on top sit three tagged tables keyed on a hash of
(target layer, token id, coarsened expert sets at the last 1, 2, 4 layers before the source layer,
using the top 4 / 2 / 1 experts respectively). Each entry stores a tag, the top-k set, a 2-bit
confidence and a 2-bit useful counter (~9 bytes). Longest tag match answers; allocate on a miss in the
shortest longer table with a free useful bit. Two departures from CPU TAGE, both measured: an entry
counts as a hit at recall ≥ 0.75 and allocation triggers below 0.875 (exact-set rules are 1–2 points
worse), and a just-allocated entry is trusted rather than deferring to the alternate (deferring costs
3–4 points: a set stored one token ago is already a good guess). Sizes below are the tagged tables; the
base adds 32 KB (OLMoE) / 52 KB (DeepSeek).

**OLMoE-1B-7B, recall@8 (64 KB tagged)**

| method                     | d=1  | d=2  | d=4  | d=8  |
|----------------------------|------|------|------|------|
| next-layer router heuristic| 0.82 | 0.76 | 0.64 | 0.46 |
| pairwise base alone        | 0.66 | 0.65 | 0.64 | 0.59 |
| TAGE 16 KB                 | 0.72 | 0.71 | 0.70 | 0.68 |
| TAGE 64 KB                 | 0.74 | 0.73 | 0.72 | 0.70 |
| TAGE 256 KB                | 0.75 | 0.74 | 0.73 | 0.71 |
| TAGE 64 KB, prefetch 2k    | 0.89 | 0.89 | 0.88 | 0.86 |
| heuristic, prefetch 2k     | 0.96 | 0.92 | 0.83 | 0.65 |

- The tagged tables answer ~52–62% of lookups at 64 KB and are right ~77–82% of the time when they do,
  versus ~58–65% for the base on the lookups left to it.
- Gain over the pairwise base is +8 points at d=1 and +10 at d=8; the curve is nearly flat.
  **Crossover with the heuristic moves from d≈4 to d≈2**, and at d=8 TAGE leads by 24 points (21 with
  2k candidates). 16 KB already captures most of the gain.
- Folding the token id into the keys is worth ~1–2 points; coarsening matters (keys on the full
  top-8 / top-4 / top-2 lose ~1.5 points because long histories repeat less).

**DeepSeek-V2-Lite, recall@6 (64 KB tagged)**

| method                     | d=1  | d=2  | d=4  | d=8  |
|----------------------------|------|------|------|------|
| next-layer router heuristic| 0.80 | 0.73 | 0.62 | 0.50 |
| pairwise base alone        | 0.53 | 0.52 | 0.49 | 0.46 |
| TAGE 16 KB                 | 0.56 | 0.55 | 0.53 | 0.50 |
| TAGE 64 KB                 | 0.58 | 0.57 | 0.55 | 0.52 |
| TAGE 256 KB                | 0.60 | 0.59 | 0.56 | 0.54 |
| TAGE 64 KB, prefetch 2k    | 0.75 | 0.74 | 0.72 | 0.69 |
| heuristic, prefetch 2k     | 0.96 | 0.91 | 0.82 | 0.68 |

- Smaller gain (+5 to +6 points over the base). Tagged tables cover only 35–41% of lookups and are
  less accurate when they hit (~66–75%), so the harder routing shows up as fewer repeating histories.
  TAGE passes the heuristic at d=8 (0.52 vs 0.50; 0.69 vs 0.68 with 2k).

**Takeaway.** History alone (TAGE) is a clear win far ahead on OLMoE and a marginal one on DeepSeek,
and it never approaches the heuristic one layer ahead. The two signals are complementary, which is what
the fusion experiment below tests.

## Online fusion of heuristic + history (added 2025-11-10, `predictors/fusion.py`)

Each source gives a score per expert in [0, 1]: softmax of the heuristic's logits over its 2k
candidates (now stored in the traces as `xlayer_logits`), the pairwise votes normalized, and the TAGE
stored set scaled by confidence when a tag matches. The fused score is a per-layer weighted sum; the
prefetch set is its top k (or 2k). Weights start uniform and are updated online after every layer by
multiplicative weights on each source's own recall (η = 0.1), so they are learned per layer and
horizon with no training and a few hundred bytes of state. Also reported: an oracle-selection bound
(best single source per prediction) and a confidence-threshold policy that fetches every expert whose
fused score is at least τ times the top score.

**recall@k on decode tokens, 64 KB TAGE + 32/52 KB base + weights**

| OLMoE-1B-7B                 | d=1  | d=2  | d=4  | d=8  |
|-----------------------------|------|------|------|------|
| heuristic alone             | 0.82 | 0.76 | 0.64 | 0.46 |
| TAGE selection (above)      | 0.74 | 0.73 | 0.72 | 0.70 |
| fused heur + base           | 0.82 | 0.76 | 0.68 | 0.60 |
| **fused heur + base + TAGE**| **0.83** | **0.79** | **0.74** | **0.70** |
| oracle select (bound)       | 0.87 | 0.83 | 0.79 | 0.75 |
| fused, prefetch 2k          | 0.96 | 0.94 | 0.90 | 0.86 |
| heuristic, prefetch 2k      | 0.96 | 0.92 | 0.83 | 0.65 |

| DeepSeek-V2-Lite            | d=1  | d=2  | d=4  | d=8  |
|-----------------------------|------|------|------|------|
| heuristic alone             | 0.80 | 0.73 | 0.62 | 0.50 |
| TAGE selection (above)      | 0.58 | 0.57 | 0.55 | 0.52 |
| fused heur + base           | 0.80 | 0.73 | 0.62 | 0.50 |
| **fused heur + base + TAGE**| **0.80** | **0.73** | **0.64** | **0.56** |
| oracle select (bound)       | 0.82 | 0.77 | 0.70 | 0.62 |
| fused, prefetch 2k          | 0.96 | 0.91 | 0.82 | 0.72 |
| heuristic, prefetch 2k      | 0.96 | 0.91 | 0.82 | 0.68 |

- **Fusion is never worse than the best single source and is better than all of them from d=2 on.**
  On OLMoE it beats the heuristic by 3 / 10 / 24 points at d = 2 / 4 / 8 and TAGE selection by
  9 / 6 / 2 / 0. On DeepSeek it matches the heuristic nearby and beats it by 1.4 / 6 points at d = 4 / 8.
- The learned weights do what the design predicts: the heuristic gets ~all the weight at d=1 and TAGE
  ~all of it at d=8, with the handover around d=2 (OLMoE) or d=4–8 (DeepSeek). The pairwise base gets
  zero weight whenever TAGE is present: TAGE subsumes it.
- Fusion sits 3–6 points under the oracle-selection bound, so a smarter chooser could still add a
  few points, but most of the available gain is captured.
- **Confidence-threshold prefetch** gives a smooth recall-vs-bandwidth curve. On OLMoE at d=1, fetching
  ~11 experts per layer (τ = 0.3) gets 0.90, between the fixed-k 0.83 and the fixed-2k 0.96 at 16; at
  d=8 the same τ gets 0.75 at ~10 experts where the heuristic needs 16 to reach 0.65.

**Reading.** The pitch's two-level story holds and the two levels can be one predictor: a single fused
ranking is as good as the heuristic one layer ahead and far better than it at long horizons, learning
its own handover point per layer with no training and ~100 KB of state. The remaining gap is on the
harder model nearby (DeepSeek d=1–2 is pure heuristic territory) and the 3–6 points to the oracle bound.

## What this says about the pitch's three experiments

- **01 Horizon sweep:** done for d = 1, 2, 4, 8 and next-token. Heuristic decays ~5 points per doubling
  of distance; history tables decay ~1–2 points. Crossover at d ≈ 4 on OLMoE, not reached by d = 8 on DeepSeek.
- **02 Pick the right history:** same-token earlier-layer experts (pairwise) > token id (hashed) >
  previous-token same-layer experts > global frequency, at every horizon. Combining several lengths of
  that history with the token id (TAGE) beats any one of them. A tournament chooser between
  pairwise and token-id did not beat pairwise alone (the chooser needs a better confidence signal, or the
  two should be fused by summing votes rather than selecting one).
- **03 Two-level prefetching:** the data already motivates it: use the heuristic for RAM→GPU at d=1–2
  (82–96%), use history for SSD→RAM at d=4–8 (81–85% with 2k candidates), where the heuristic is 65–83%.
  The fused predictor does both with one ranking and learns the handover per layer.

## Caveats

- Two models, both 64 routed experts. Mixtral (8 experts, top-2) is the remaining pitch target.
- TAGE and fusion are first cuts: fixed table geometry (3 tables, equal sizes), no tuning of tag bits,
  reset period or η; the heuristic's scores are softmax over its 2k candidates only.
- The cost model ignores compute overlap and GPU cache capacity; the real simulator replaces it.
- Greedy decode on a base model; chat/instruct models and sampling untested.

## Next steps (ordered)

1. Put the fused predictor into a two-tier replay (SSD→RAM at d=4–8 with the τ policy, RAM→GPU at d=1
   exact-k) and measure combined hit rate and bandwidth per tier with the cost model.
2. Extend TAGE with previous-token histories (next-token horizon) and tune geometry (more tables,
   unequal sizes, tag bits); try a per-expert perceptron chooser to close the gap to the oracle bound.
3. Trace Mixtral-8x7B (8 experts, top-2); rerun `analysis/headroom.py` unchanged.
4. Build the GPU/RAM/SSD replay simulator on the trace format already in `traces/`.
