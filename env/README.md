# Environment: conda env `moe-bp` (Python 3.12)

Created 2025-11-06. Lives at `/scr/cportner/miniconda3/envs/moe-bp`.

```bash
conda activate moe-bp
export HF_HOME=/scr/cportner/hf     # model + dataset cache on /scr (root disk is nearly full)
```

Recreate from scratch:
```bash
conda create -n moe-bp python=3.12
conda activate moe-bp
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -r env/requirements.txt
```

Notes
- transformers is 5.x here; routers are custom modules returning (logits, weights, indices), and
  `collect/collect_traces.py` handles both that and the 4.x nn.Linear gate.
- Gated models (Mixtral): the HF token is looked up under `$HF_HOME/token`, so either copy
  `~/.cache/huggingface/token` there, run `hf auth login` inside this env, or export `HF_TOKEN`.
