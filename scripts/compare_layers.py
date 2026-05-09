"""
Layer-by-layer comparison between vanilla PyTorch and the ZK pipeline.
Prints per-layer L1/max error to identify where numerical divergence begins.

Usage:
    python compare_layers.py inputs.json --num_layers 4
"""

import os, json, argparse

_orig_system = os.system
os.system = lambda cmd: _orig_system(cmd + ' > /dev/null 2>&1')
import torch
import numpy as np
from fileio_utils import save_int
from full_run import run_layer, setup, SCALING_LOG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()

    tokenizer, model, embed_dim, hidden_dim, num_layers, workdir = setup(args.model_size)

    with open(args.input_file) as f:
        text = json.load(f)[0]

    token_ids = tokenizer(text, return_tensors='pt')['input_ids']
    seq_len = token_ids.shape[1]
    print(f'seq_len={seq_len}, embed_dim={embed_dim}')

    # Vanilla PyTorch: capture per-layer outputs via hooks
    vanilla = {}
    handles = []
    for i in range(args.num_layers):
        def hook(module, inp, out, i=i):
            vanilla[i] = out[0].detach().float().squeeze(0)
        handles.append(model.model.layers[i].register_forward_hook(hook))
    with torch.no_grad():
        model(token_ids)
    for h in handles:
        h.remove()

    # ZK pipeline: run layer by layer
    with torch.no_grad():
        embedding = model.model.embed_tokens(token_ids).squeeze(0).float()
    save_int(embedding, 1 << SCALING_LOG, '_embed.bin')

    print(f'\n{"Layer":>5}  {"L1 err":>10}  {"Max err":>10}  {"Rel L1":>10}')
    print('-' * 45)

    for i in range(args.num_layers):
        inp = '_embed.bin' if i == 0 else f'_cmp_layer_{i-1}.bin'
        out = f'_cmp_layer_{i}.bin'
        run_layer(model.model.layers[i], i, seq_len, embed_dim, hidden_dim, workdir, inp, out)

        zk = torch.tensor(
            np.fromfile(out, dtype=np.int32).reshape(seq_len, embed_dim),
            dtype=torch.float32
        ) / (1 << SCALING_LOG)

        diff = (zk - vanilla[i].cpu()).abs()
        l1   = diff.mean().item()
        mx   = diff.max().item()
        rel  = l1 / vanilla[i].abs().mean().item()
        print(f'{i:>5}  {l1:>10.5f}  {mx:>10.5f}  {rel:>10.4%}')

    for i in range(args.num_layers):
        f = f'_cmp_layer_{i}.bin'
        if os.path.exists(f): os.remove(f)
    if os.path.exists('_embed.bin'): os.remove('_embed.bin')
