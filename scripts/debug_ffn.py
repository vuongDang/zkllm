"""
Debug FFN sub-steps: compare committed-weight Python computation vs vanilla PyTorch.
Runs for a specific layer on the ffn_in already saved to disk.

Usage:
    python debug_ffn.py --layer 1   # after running debug_layer.py --layer 1 once
    python debug_ffn.py --layer 0
"""
import os, json, argparse, math
import torch
import numpy as np
from fileio_utils import save_int, load_int
from full_run import setup, SCALING_LOG, SWIGLU_TABLE
import sys

_orig = os.system
os.system = lambda cmd: _orig(cmd + ' > /dev/null 2>&1')

WORKDIR = './zkllm-workdir/Llama-2-7b'


def load_committed_weight(path, dtype=torch.float32):
    w = np.fromfile(path, dtype=np.int32)
    return torch.tensor(w, dtype=dtype) / (1 << SCALING_LOG)


def swiglu_python(x_float):
    """x * sigmoid(x) — the SwiGLU activation."""
    return x_float * torch.sigmoid(x_float)


def run_ffn_debug(layer, layer_idx, seq_len, embed_dim, hidden_dim, ffn_in_file):
    lp = f'layer-{layer_idx}'
    print(f'\n=== FFN sub-step debug for layer {layer_idx} ===')

    # Load ffn_in
    ffn_in_int32 = np.fromfile(ffn_in_file, dtype=np.int32).reshape(seq_len, embed_dim)
    ffn_in_float = torch.tensor(ffn_in_int32, dtype=torch.float32) / (1 << SCALING_LOG)

    # Vanilla sub-step: apply each weight manually with full precision
    with torch.no_grad():
        up_van   = layer.mlp.up_proj(ffn_in_float)
        gate_van = layer.mlp.gate_proj(ffn_in_float)
        swiglu_van = swiglu_python(gate_van) * up_van
        down_van = layer.mlp.down_proj(swiglu_van)

    # Load committed weights (stored as w.T: shape is (in_dim, out_dim))
    # up_proj vanilla: (hidden, embed) → committed: (embed, hidden)
    # down_proj vanilla: (embed, hidden) → committed: (hidden, embed)
    up_w    = load_committed_weight(f'{WORKDIR}/{lp}-mlp.up_proj.weight-int.bin')   .reshape(embed_dim, hidden_dim)
    gate_w  = load_committed_weight(f'{WORKDIR}/{lp}-mlp.gate_proj.weight-int.bin') .reshape(embed_dim, hidden_dim)
    down_w  = load_committed_weight(f'{WORKDIR}/{lp}-mlp.down_proj.weight-int.bin') .reshape(hidden_dim, embed_dim)

    # Simulate ZK FFN in Python with committed weights
    # ZK computes: output = X @ W  (W has shape (in, out))
    up_zk   = ffn_in_float @ up_w         # (seq, hidden) — float
    gate_zk = ffn_in_float @ gate_w       # (seq, hidden) — float

    # Rescale up to scale 2^16: quantize then dequantize
    up_zk_q   = torch.round(up_zk   * (1 << SCALING_LOG)).clamp(-2**31, 2**31-1).to(torch.int32).float() / (1 << SCALING_LOG)
    gate_zk_q = torch.round(gate_zk * (1 << SCALING_LOG)).clamp(-2**31, 2**31-1).to(torch.int32).float() / (1 << SCALING_LOG)

    swiglu_zk   = swiglu_python(gate_zk_q) * up_zk_q
    swiglu_zk_q = torch.round(swiglu_zk * (1 << SCALING_LOG)).clamp(-2**31, 2**31-1).to(torch.int32).float() / (1 << SCALING_LOG)

    down_zk   = swiglu_zk_q @ down_w
    down_zk_q = torch.round(down_zk * (1 << SCALING_LOG)).clamp(-2**31, 2**31-1).to(torch.int32).float() / (1 << SCALING_LOG)

    def cmp(name, zk, van):
        d = (zk - van).abs()
        print(f'  {name:35s}  L1={d.mean():.5f}  Max={d.max():.5f}')
        print(f'       zk  abs-max={zk.abs().max():.3f}  van abs-max={van.abs().max():.3f}')

    cmp('up_proj output (before rescale)',  up_zk,   up_van)
    cmp('up_proj output (after rescale)',   up_zk_q, up_van)
    cmp('gate_proj output (before rescale)', gate_zk,   gate_van)
    cmp('gate_proj output (after rescale)',  gate_zk_q, gate_van)
    cmp('swiglu output (before rescale)',   swiglu_zk,   swiglu_van)
    cmp('swiglu output (after rescale)',    swiglu_zk_q, swiglu_van)
    cmp('down_proj output (before rescale)', down_zk,   down_van)
    cmp('down_proj output (after rescale)',  down_zk_q, down_van)

    # Additional: check max absolute values in vanilla outputs at each stage
    print(f'\n  Vanilla magnitudes:')
    print(f'    up    max={up_van.abs().max():.3f}')
    print(f'    gate  max={gate_van.abs().max():.3f}')
    print(f'    silu  max={swiglu_van.abs().max():.3f}')
    print(f'    down  max={down_van.abs().max():.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7)
    parser.add_argument('--layer', type=int, default=1)
    args = parser.parse_args()

    tokenizer, model, embed_dim, hidden_dim, num_layers, workdir = setup(args.model_size)

    with open(args.input_file) as f:
        text = json.load(f)[0]

    token_ids = tokenizer(text, return_tensors='pt')['input_ids']
    seq_len = token_ids.shape[1]
    print(f'seq_len={seq_len}, layer={args.layer}')

    # Run ZK pipeline up through post-attn RMSNorm for the target layer
    from full_run import run_layer
    with torch.no_grad():
        embedding = model.model.embed_tokens(token_ids).squeeze(0).float()
    save_int(embedding, 1 << SCALING_LOG, '_ffndbg_embed.bin')

    for i in range(args.layer):
        inp = '_ffndbg_embed.bin' if i == 0 else f'_ffndbg_prev_{i-1}.bin'
        out = f'_ffndbg_prev_{i}.bin'
        run_layer(model.model.layers[i], i, seq_len, embed_dim, hidden_dim, workdir, inp, out)

    # Now run up through the post-attn RMSNorm for the target layer
    lp = f'layer-{args.layer}'
    input_f   = '_ffndbg_embed.bin' if args.layer == 0 else f'_ffndbg_prev_{args.layer - 1}.bin'
    attn_in_f = '_ffndbg_attn_in.bin'
    attn_out_f= '_ffndbg_attn_out.bin'
    post_attn_f='_ffndbg_post_attn.bin'
    ffn_in_f  = '_ffndbg_ffn_in.bin'
    layer     = model.model.layers[args.layer]

    X = torch.tensor(np.fromfile(input_f, dtype=np.int32).reshape(seq_len, embed_dim),
                     dtype=torch.float64, device=0) / (1 << SCALING_LOG)
    rms_inv = 1 / torch.sqrt(torch.mean(X**2, dim=1) + layer.input_layernorm.variance_epsilon)
    save_int(rms_inv.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm input {input_f} {seq_len} {embed_dim} {workdir} {lp} {attn_in_f}')
    os.remove('rms_inv_temp.bin')

    os.system(f'./self-attn linear {attn_in_f} {seq_len} {embed_dim} {workdir} {lp} {attn_out_f}')
    Q = load_int('temp_Q.bin').reshape(seq_len, embed_dim).float() / (1 << 16)
    K = load_int('temp_K.bin').reshape(seq_len, embed_dim).float() / (1 << 16)
    V = load_int('temp_V.bin').reshape(seq_len, embed_dim).float() / (1 << 16)

    from full_run import rotate_half, ACCU_LOGSF, VALUE_LOGSF
    from fileio_utils import to_int64, to_float, fromto_int64
    num_heads = layer.self_attn.num_heads
    head_dim  = layer.self_attn.head_dim
    Q = Q.view(seq_len, num_heads, head_dim).transpose(0, 1)
    K = K.view(seq_len, num_heads, head_dim).transpose(0, 1)
    V = V.view(seq_len, num_heads, head_dim).transpose(0, 1)
    layer.self_attn.rotary_emb.to(0)
    cos, sin = layer.self_attn.rotary_emb(Q.float(), seq_len=seq_len)
    cos = cos.squeeze(0).squeeze(0)
    sin = sin.squeeze(0).squeeze(0)
    Q = (Q * cos.unsqueeze(0) + rotate_half(Q) * sin.unsqueeze(0)).to(torch.float64)
    K = (K * cos.unsqueeze(0) + rotate_half(K) * sin.unsqueeze(0)).to(torch.float64)
    A = to_int64(Q @ K.transpose(-2, -1), ACCU_LOGSF)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=0, dtype=bool), diagonal=1)
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    import math
    shift = math.sqrt(head_dim) * torch.log(
        (torch.exp(to_float(A, ACCU_LOGSF) / math.sqrt(head_dim)) * ~mask).sum(dim=-1, keepdim=True))
    A -= to_int64(shift, ACCU_LOGSF)
    attn_weights = torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(head_dim)).float() * ~mask
    attn_v = fromto_int64(attn_weights @ V, VALUE_LOGSF)
    save_int(attn_v.transpose(0, 1).contiguous().view(seq_len, embed_dim).float(), 1 << VALUE_LOGSF, 'temp_attn_out.bin')
    os.system(f'./self-attn attn {attn_in_f} {seq_len} {embed_dim} {workdir} {lp} {attn_out_f}')
    os.system('rm -f ./temp_Q.bin ./temp_K.bin ./temp_V.bin ./temp_attn_out.bin')
    attn_v_flat = attn_v.float().transpose(0, 1).contiguous().view(seq_len, embed_dim).cpu()
    with torch.no_grad():
        attn_out_tensor = layer.self_attn.o_proj(attn_v_flat)
    save_int(attn_out_tensor, 1 << SCALING_LOG, attn_out_f)
    os.system(f'./skip-connection {input_f} {attn_out_f} {post_attn_f}')

    X2 = torch.tensor(np.fromfile(post_attn_f, dtype=np.int32).reshape(seq_len, embed_dim),
                      dtype=torch.float64, device=0) / (1 << SCALING_LOG)
    rms_inv2 = 1 / torch.sqrt(torch.mean(X2**2, dim=1) + layer.post_attention_layernorm.variance_epsilon)
    save_int(rms_inv2.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm post_attention {post_attn_f} {seq_len} {embed_dim} {workdir} {lp} {ffn_in_f}')
    os.remove('rms_inv_temp.bin')

    run_ffn_debug(layer, args.layer, seq_len, embed_dim, hidden_dim, ffn_in_f)

    for f in ['_ffndbg_embed.bin', attn_in_f, attn_out_f, post_attn_f, ffn_in_f] + \
             [f'_ffndbg_prev_{i}.bin' for i in range(args.layer)]:
        if os.path.exists(f):
            os.remove(f)
