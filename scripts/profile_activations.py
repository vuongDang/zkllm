"""
Profile activation ranges across LLaMA-2 layers using vanilla PyTorch.
Use this to determine safe scaling factors and swiglu table range for the ZK pipeline.

Usage:
    python profile_activations.py inputs.json
    python profile_activations.py inputs.json --model_size 7 --percentile 99.9

Output: activation_ranges.json with per-layer min/max/percentiles for:
  - layer_out        : residual stream output (determines SCALING_LOG)
  - gate_pre_swiglu  : gate projection output (determines swiglu table range)
  - attn_out         : attention output before skip connection
  - ffn_out          : FFN output before skip connection
"""

# use WikiText-2 (200 sequences of 256 tokens — much more representative)                   
  #conda run -n zkllm-env pip install datasets                                                 
  #conda run -n zkllm-env python profile_activations.py --num_samples 200      

import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_wikitext(tokenizer, num_samples, seq_len):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # join all text, split into chunks of seq_len tokens
    full_text = '\n'.join(ds['text'])
    token_ids = tokenizer(full_text, return_tensors='pt')['input_ids'][0]
    chunks = [token_ids[i:i+seq_len] for i in range(0, len(token_ids) - seq_len, seq_len)]
    return [tokenizer.decode(c) for c in chunks[:num_samples]]


def profile(model, tokenizer, prompts, percentile):
    # layer_idx -> name -> list of observed values (sampled for memory)
    buckets = {}

    def make_hook(layer_idx, name):
        def hook(module, inp, out):
            t = out.detach().float() if isinstance(out, torch.Tensor) else out[0].detach().float()
            key = (layer_idx, name)
            if key not in buckets:
                buckets[key] = []
            # flatten and subsample to keep memory bounded (~10k values per key)
            vals = t.flatten().cpu().numpy()
            if len(vals) > 10_000:
                vals = np.random.choice(vals, 10_000, replace=False)
            buckets[key].append(vals)
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i, 'layer_out')))
        handles.append(layer.mlp.gate_proj.register_forward_hook(make_hook(i, 'gate_pre_swiglu')))
        handles.append(layer.mlp.register_forward_hook(make_hook(i, 'ffn_out')))
        handles.append(layer.self_attn.register_forward_hook(make_hook(i, 'attn_out')))

    with torch.no_grad():
        for text in prompts:
            token_ids = tokenizer(text, return_tensors='pt')['input_ids']
            model(token_ids)

    for h in handles:
        h.remove()

    lo = (100 - percentile) / 2
    hi = 100 - lo
    results = {}
    for (layer_idx, name), chunks in sorted(buckets.items()):
        all_vals = np.concatenate(chunks)
        results.setdefault(layer_idx, {})[name] = {
            'min':    float(all_vals.min()),
            'max':    float(all_vals.max()),
            f'p{lo:.3g}':  float(np.percentile(all_vals, lo)),
            f'p{hi:.3g}': float(np.percentile(all_vals, hi)),
        }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='?', help='JSON file with list of prompts (omit to use WikiText-2)')
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7)
    parser.add_argument('--percentile', type=float, default=99.9,
                        help='Percentile band to report (default 99.9 → p0.05 and p99.95)')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of WikiText-2 sequences to use when no input_file given')
    parser.add_argument('--seq_len', type=int, default=256,
                        help='Sequence length for WikiText-2 chunks')
    args = parser.parse_args()

    model_card = f'meta-llama/Llama-2-{args.model_size}b-hf'
    print(f'Loading {model_card}...')
    tokenizer = AutoTokenizer.from_pretrained(model_card, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only=True)
    model.eval()

    if args.input_file:
        with open(args.input_file) as f:
            prompts = json.load(f)
    else:
        print(f'Loading WikiText-2 ({args.num_samples} × {args.seq_len} tokens)...')
        prompts = load_wikitext(tokenizer, args.num_samples, args.seq_len)
    print(f'Profiling {len(prompts)} prompt(s)...')

    results = profile(model, tokenizer, prompts, args.percentile)

    out_path = 'activation_ranges.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {out_path}')

    # Print summary: worst-case ranges across all layers per activation type
    names = ['gate_pre_swiglu', 'layer_out', 'attn_out', 'ffn_out']
    print(f'\n{"Activation":20s}  {"abs_max":>10}  {"note"}')
    print('-' * 60)
    abs_maxes = {}
    for name in names:
        abs_maxes[name] = max(
            max(abs(v['min']), abs(v['max']))
            for layer in results.values()
            if name in layer
            for v in [layer[name]]
        )
    # safe SCALING_LOG must account for all values written to int32 by the binaries
    int32_activations = [abs_maxes[n] for n in ['layer_out', 'attn_out', 'ffn_out']]
    global_safe_log = int(np.floor(np.log2(2**31 / max(int32_activations))))
    for name in names:
        abs_max = abs_maxes[name]
        note = ''
        if name == 'gate_pre_swiglu':
            note = f'← swiglu table must cover at least ±{abs_max:.0f}'
        elif name == 'ffn_out':
            note = f'← largest intermediate → drives SCALING_LOG'
        print(f'{name:20s}  {abs_max:>10.2f}  {note}')
    print(f'\n→ recommended SCALING_LOG = {global_safe_log}  (safe for all activations)')

  