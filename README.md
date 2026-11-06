# moe-bp: expert "branch prediction" for MoEs — trace study

Preliminary trace testing for the 828 project (see `../828 project pitch.pdf` and
`../preliminary-testing-plan.md`).

## Layout
- `collect/collect_traces.py` — hooks every router Linear, runs prefill + greedy decode at batch 1,
  records full router logits per token/layer plus the Fate/HOBBIT-style "push layer l's input through
  layer l+d's router" prediction for d = 1..8. One `.npz` per (model, workload).
- `analysis/headroom.py` — headroom study over traces: usage skew, LRU caching, previous-token reuse,
  cross-layer router heuristic vs horizon, online counter-table predictors (bimodal, token-id, pairwise
  correlating, tournament), crude PCIe cost model, plots.
- `traces/` — collected traces (not committed). `results/` — logs, JSON, PNGs.
- `env/` — conda env `moe-bp` (Python 3.12, transformers 5.x) and pinned requirements. Prompts are read
  straight from the datasets' parquet files via `huggingface_hub`, which works in any env.

## Run
```bash
conda activate moe-bp               # see env/README.md
export HF_HOME=/scr/cportner/hf
python collect/collect_traces.py --workload wikitext  --n-prompts 60 --decode-tokens 160 --device cuda:0
python collect/collect_traces.py --workload gsm8k     --n-prompts 60 --decode-tokens 160 --device cuda:1
python collect/collect_traces.py --workload humaneval --n-prompts 60 --decode-tokens 160 --device cuda:2
python analysis/headroom.py            # all OLMoE traces -> results/headroom_OLMoE-1B-7B-0924.{json,png}
```

## Metric
recall@k = |predicted set ∩ true top-k| / k, averaged over decode tokens only. Prefill tokens still
flow through caches and tables in order so the state is realistic.
