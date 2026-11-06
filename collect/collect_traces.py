"""Collect MoE routing traces.

For every token and every MoE layer we record:
  * full router logits (float16)                       -> offline top-k, margins, entropy
  * "cross-layer" heuristic predictions: the router input x of layer l pushed
    through the router weights of layers l+1..l+H (Fate/HOBBIT-style) -> top-2k ids
  * token id, sequence id, position, phase (0 = prefill, 1 = decode)

Decode is a manual greedy loop at batch size 1 so each forward pass == one token
and the hook bookkeeping is trivial.
"""
import argparse, json, os, time, sys
import numpy as np
import torch
import torch.nn as nn



def find_routers(model, num_experts):
    """Return [(layer_idx, module)] for router Linear layers, ordered by layer."""
    found = []
    for name, m in model.named_modules():
        last = name.split(".")[-1]
        w = getattr(m, "weight", None)
        is_router = (isinstance(m, nn.Linear) and m.out_features == num_experts and ("gate" in last or "router" in name)) or \
                    (not isinstance(m, nn.Linear) and w is not None and w.dim() == 2 and w.shape[0] == num_experts and last in ("gate", "router"))
        if is_router:
            # pull layer index from the name, e.g. model.layers.3.mlp.gate
            parts = name.split(".")
            idx = next(int(p) for p in parts if p.isdigit())
            found.append((idx, name, m))
    found.sort(key=lambda t: t[0])
    return found


class Recorder:
    def __init__(self, routers, horizon, topk2):
        self.routers = routers          # list of (layer_idx, name, module)
        self.L = len(routers)
        self.H = horizon
        self.topk2 = topk2
        self.W = [r[2].weight.detach() for r in routers]  # [E, hidden] each
        self.reset()

    def reset(self):
        self.step_logits = [None] * self.L
        self.step_xlayer = [None] * self.L

    def make_hook(self, li):
        def hook(mod, inp, out):
            x = inp[0]
            x2 = x.reshape(-1, x.shape[-1])
            E = self.W[li].shape[0]
            idx = None
            if isinstance(out, (tuple, list)):          # transformers>=5 routers: (router_logits, weights, indices)
                floats = [t for t in out if torch.is_tensor(t) and t.is_floating_point() and t.shape[-1] == E]
                ints = [t for t in out if torch.is_tensor(t) and not t.is_floating_point()]
                logits = floats[0].reshape(-1, E) if floats else x2.float() @ self.W[li].float().T
                idx = ints[0].reshape(-1, ints[0].shape[-1]) if ints else None
            elif torch.is_tensor(out) and out.shape[-1] == E:   # nn.Linear gate (transformers 4.x)
                logits = out.reshape(-1, E)
            else:                                        # unknown gate: recompute from its weight
                logits = x2.float() @ self.W[li].float().T
            self.step_logits[li] = logits.detach().to(torch.float16).cpu()
            if not getattr(self, "_checked", False) and idx is not None:
                # offline top-k of the stored logits must equal the experts the model actually used
                mine = logits.float().topk(idx.shape[-1], dim=-1).indices
                assert torch.equal(mine.sort(-1).values, idx.to(mine.device).sort(-1).values), "stored logits do not reproduce the router's expert choice"
                self._checked = True
            xl = torch.full((x2.shape[0], self.H, self.topk2), -1, dtype=torch.int16)
            for d in range(1, self.H + 1):
                if li + d < self.L:
                    lg = x2.float() @ self.W[li + d].float().T
                    xl[:, d - 1, :] = lg.topk(self.topk2, dim=-1).indices.to(torch.int16).cpu()
            self.step_xlayer[li] = xl
        return hook

    def attach(self):
        self.handles = [r[2].register_forward_hook(self.make_hook(i)) for i, r in enumerate(self.routers)]

    def pop(self):
        """Return (logits [T,L,E], xlayer [T,L,H,2k]) for the last forward and clear."""
        lg = torch.stack(self.step_logits, dim=1)
        xl = torch.stack(self.step_xlayer, dim=1)
        self.reset()
        return lg, xl


def load_prompts(workload, n, tok, max_prompt_tokens):
    """Return list of prompt strings, read straight from the datasets' parquet files
    (the `datasets` library is broken under Python 3.14 pickling)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    files = {
        "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet"),
        "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
        "humaneval": ("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"),
    }
    repo, fn = files[workload]
    path = hf_hub_download(repo, fn, repo_type="dataset")
    df = pd.read_parquet(path)
    prompts = []
    if workload == "wikitext":
        for t in df["text"]:
            t = t.strip()
            if len(t) > 600 and not t.startswith("="):
                prompts.append(t)
            if len(prompts) >= n: break
    elif workload == "gsm8k":
        prompts = ["Question: " + q.strip() + "\nAnswer: Let's think step by step." for q in df["question"][:n]]
    elif workload == "humaneval":
        prompts = list(df["prompt"][:n])
    out = []
    for p in prompts[:n]:
        ids = tok(p, return_tensors="pt").input_ids[0]
        if ids.numel() > max_prompt_tokens:
            ids = ids[:max_prompt_tokens]
            p = tok.decode(ids, skip_special_tokens=True)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--workload", default="wikitext", choices=["wikitext", "gsm8k", "humaneval"])
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--max-prompt-tokens", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    E = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None) or getattr(cfg, "n_routed_experts")
    K = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "num_experts_per_token")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    routers = find_routers(model, E)
    print(f"model={args.model} experts={E} topk={K} moe_layers={len(routers)} first={routers[0][1]}")
    rec = Recorder(routers, args.horizon, 2 * K)
    rec.attach()

    prompts = load_prompts(args.workload, args.n_prompts, tok, args.max_prompt_tokens)
    print(f"workload={args.workload} prompts={len(prompts)}")

    all_logits, all_xl, tok_ids, seq_ids, pos_ids, phases = [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for si, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.to(args.device)
            out = model(input_ids=ids, use_cache=True)
            lg, xl = rec.pop()
            T = ids.shape[1]
            all_logits.append(lg); all_xl.append(xl)
            tok_ids.append(ids[0].cpu()); seq_ids.append(torch.full((T,), si)); pos_ids.append(torch.arange(T)); phases.append(torch.zeros(T, dtype=torch.int8))
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            for t in range(args.decode_tokens):
                out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
                lg, xl = rec.pop()
                all_logits.append(lg); all_xl.append(xl)
                tok_ids.append(nxt.cpu()); seq_ids.append(torch.tensor([si])); pos_ids.append(torch.tensor([T + t])); phases.append(torch.ones(1, dtype=torch.int8))
                past = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1)
                if nxt.item() == tok.eos_token_id:
                    break
            if si % 5 == 0:
                n = sum(x.shape[0] for x in all_logits)
                print(f"  seq {si}/{len(prompts)} tokens={n} elapsed={time.time()-t0:.0f}s", flush=True)

    logits = torch.cat(all_logits).numpy()
    xlayer = torch.cat(all_xl).numpy()
    meta = dict(model=args.model, workload=args.workload, num_experts=int(E), topk=int(K), num_moe_layers=len(routers),
                horizon=args.horizon, n_prompts=len(prompts), decode_tokens=args.decode_tokens,
                router_names=[r[1] for r in routers], n_tokens=int(logits.shape[0]))
    out_path = args.out or f"traces/{args.model.split('/')[-1]}__{args.workload}.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, logits=logits, xlayer=xlayer,
                        token_id=torch.cat(tok_ids).numpy().astype(np.int32), seq_id=torch.cat(seq_ids).numpy().astype(np.int32),
                        pos=torch.cat(pos_ids).numpy().astype(np.int32), phase=torch.cat(phases).numpy(), meta=json.dumps(meta))
    print(f"saved {out_path} tokens={logits.shape[0]} layers={logits.shape[1]} experts={logits.shape[2]} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
