# moe-bp: expert "branch prediction" for MoEs — trace study

Preliminary trace testing for the 828 project.

## Layout
- `collect/collect_traces.py` — hooks every router Linear, runs prefill + greedy decode at batch 1,
  records full router logits per token/layer plus the Fate/HOBBIT-style "push layer l's input through
  layer l+d's router" prediction for d = 1..8. One `.npz` per (model, workload).
- `analysis/headroom.py` — headroom study over traces: usage skew, LRU caching, previous-token reuse,
  cross-layer router heuristic vs horizon, online counter-table predictors (bimodal, token-id, pairwise
  correlating, tournament), crude PCIe cost model, plots.
- `predictors/tage.py` — TAGE-style predictor: pairwise base table plus tagged tables keyed on hashed
  (token id, coarsened expert sets at 1/2/4 earlier layers); longest tag match wins, allocate on miss,
  confidence + useful bits. Sweeps sizes and horizons in parallel and reports provider share/recall.
- `predictors/fusion.py` — online fusion: ranks experts by a per-layer weighted sum of heuristic softmax,
  pairwise votes and TAGE stored sets; weights learned online (multiplicative, no training). Reports
  singles, an oracle-selection bound, and a confidence-threshold prefetch curve (recall vs experts fetched).
- `traces/` — collected traces (gitignored). `results/` — analysis JSON, plots, and text logs.
- `env/` — conda environment spec and pinned requirements. Prompts are read straight from the
  datasets' parquet files via `huggingface_hub`.

## Setup
See `env/README.md` for the conda environment. Model and dataset downloads go through the Hugging Face
cache; point `HF_HOME` at a disk with room (~13 GB for OLMoE, ~30 GB for DeepSeek-V2-Lite) if your home
directory is small. Gated models need a Hugging Face token (`hf auth login` or `export HF_TOKEN=...`).

## Run
```bash
conda activate moe-bp
# one GPU with >= 16 GB is enough for OLMoE; DeepSeek-V2-Lite needs ~32 GB. Use --device to pick a GPU.
python collect/collect_traces.py --workload wikitext  --n-prompts 60 --decode-tokens 160
python collect/collect_traces.py --workload gsm8k     --n-prompts 60 --decode-tokens 160
python collect/collect_traces.py --workload humaneval --n-prompts 60 --decode-tokens 160
python analysis/headroom.py            # all OLMoE traces -> results/headroom_OLMoE-1B-7B-0924.{json,png}

python collect/collect_traces.py --model deepseek-ai/DeepSeek-V2-Lite --workload gsm8k --n-prompts 40 --decode-tokens 128
python analysis/headroom.py --traces traces/DeepSeek-V2-Lite__*.npz --expert-mb 17.3

python predictors/tage.py --sizes-kb 16 64 256                      # OLMoE, default tables 1:4,2:2,4:1
python predictors/tage.py --traces traces/DeepSeek-V2-Lite__*.npz --no-token --tables 1:8,2:4,4:2 --tag fine
python predictors/tage.py --hit-thresh 1.0 --alloc-thresh 1.0 --alt-on-na --tag strict   # CPU-TAGE rules (worse here)

python predictors/fusion.py                                          # OLMoE: heur+base, heur+base+tage, base+tage
python predictors/fusion.py --traces traces/DeepSeek-V2-Lite__*.npz --eta 0.1
```
Traces must contain `xlayer_logits` (collector versions from 2025-11-10 on) for `fusion.py`.
Runs for different workloads are independent, so they can go on separate GPUs in parallel.
Collected traces land in `traces/` (gitignored, ~50 MB per model-workload). Findings so far: `findings.md`.

## Metric
recall@k = |predicted set ∩ true top-k| / k, averaged over decode tokens only. Prefill tokens still
flow through caches and tables in order so the state is realistic.
