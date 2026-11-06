# Environment

Conda env `moe-bp`: Python 3.12, PyTorch 2.15 (CUDA 13.2 wheels), transformers 5.22. Pinned versions in
`requirements.txt`.

```bash
conda create -n moe-bp python=3.12
conda activate moe-bp
pip install torch --index-url https://download.pytorch.org/whl/cu132   # pick the CUDA tag your driver supports
pip install -r env/requirements.txt
```

Optional but recommended:
```bash
export HF_HOME=/path/with/space      # Hugging Face model + dataset cache (tens of GB)
hf auth login                        # only needed for gated models such as Mixtral
```

Notes
- transformers 5.x routers are custom modules returning (logits, weights, indices); 4.x used a plain
  `nn.Linear` gate. `collect/collect_traces.py` handles both, and self-checks that the stored logits
  reproduce the experts the model actually used.
- Python 3.14 is not supported: the `datasets` library fails there, and the transformers / huggingface_hub
  version pins conflict.
