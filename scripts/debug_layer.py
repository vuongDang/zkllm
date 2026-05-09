"""
Step-by-step comparison of a single layer between vanilla PyTorch and the ZK pipeline.
Prints L1/max error at each sub-component to identify where the ~1482 error originates.

Usage:
    python debug_layer.py inputs.json --layer 1
"""

import os, json, argparse, math
import torch
import numpy as np
from fileio_utils import save_int, load_int, to_int64, to_float, fromto_int64
from full_run import setup, SCALING_LOG, VALUE_LOGSF, ACCU_LOGSF, rotate_half

_orig_system = os.system
os.system = lambda cmd: _orig_system(cmd + ' > /dev/null 2>&1')

WORKDIR = './zkllm-workdir/Llama-2-7b'


def cmp(label, zk_int32, vanilla_float, scale=SCALING_LOG):
    zk = zk_int32.float() / (1 << scale)
    diff = (zk - vanilla_float.cpu()).abs()
    l1 = diff.mean().item()
    mx = diff.max().item()
    print(f'  {label:35s}  L1={l1:.5f}  Max={mx:.5f}')
    return mx


def run_step_by_step(layer, layer_idx, seq_len, embed_dim, hidden_dim, input_file, model):
    lp = f'layer-{layer_idx}'
    attn_in_f   = f'_dbg_attn_in.bin'
    attn_out_f  = f'_dbg_attn_out.bin'
    post_attn_f = f'_dbg_post_attn.bin'
    ffn_in_f    = f'_dbg_ffn_in.bin'
    ffn_out_f   = f'_dbg_ffn_out.bin'
    output_f    = f'_dbg_out.bin'

    # ── Vanilla hooks ──────────────────────────────────────────────────
    captured = {}

    def hook_norm_in(m, inp, out):
        captured['after_input_norm'] = out.detach().float().squeeze(0)

    def hook_attn(m, inp, out):
        captured['attn_out_pre_res'] = out[0].detach().float().squeeze(0)

    def hook_norm_post(m, inp, out):
        captured['after_post_norm'] = out.detach().float().squeeze(0)

    def hook_mlp(m, inp, out):
        captured['mlp_out_pre_res'] = out.detach().float().squeeze(0)

    def hook_layer(m, inp, out):
        captured['layer_out'] = out[0].detach().float().squeeze(0)

    handles = [
        layer.input_layernorm.register_forward_hook(hook_norm_in),
        layer.self_attn.register_forward_hook(hook_attn),
        layer.post_attention_layernorm.register_forward_hook(hook_norm_post),
        layer.mlp.register_forward_hook(hook_mlp),
        layer.register_forward_hook(hook_layer),
    ]

    # load input (divide by scale to get true float values)
    X_int32 = np.fromfile(input_file, dtype=np.int32).reshape(seq_len, embed_dim)
    X_float = torch.tensor(X_int32, dtype=torch.float32) / (1 << SCALING_LOG)
    with torch.no_grad():
        layer(X_float.unsqueeze(0))
    for h in handles:
        h.remove()

    # after_attn_skip = input + attn_out_pre_res
    captured['after_attn_skip'] = (X_float + captured['attn_out_pre_res']).cpu()

    print(f'\n=== Layer {layer_idx} sub-component errors ===')

    # ── ZK step 1: input RMSNorm ────────────────────────────────────────
    X_f64 = torch.tensor(X_int32, dtype=torch.float64, device=0) / (1 << SCALING_LOG)
    rms_inv = 1 / torch.sqrt(torch.mean(X_f64 ** 2, dim=1) + layer.input_layernorm.variance_epsilon)
    save_int(rms_inv.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm input {input_file} {seq_len} {embed_dim} {WORKDIR} {lp} {attn_in_f}')
    os.remove('rms_inv_temp.bin')
    cmp('1. after input RMSNorm',
        torch.from_numpy(np.fromfile(attn_in_f, dtype=np.int32).reshape(seq_len, embed_dim)),
        captured['after_input_norm'])

    # ── ZK step 2: Q, K, V projections ──────────────────────────────────
    os.system(f'./self-attn linear {attn_in_f} {seq_len} {embed_dim} {WORKDIR} {lp} {attn_out_f}')
    Q = load_int('temp_Q.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    K = load_int('temp_K.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    V = load_int('temp_V.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)

    with torch.no_grad():
        Q_van = layer.self_attn.q_proj(captured['after_input_norm']).cpu()
        K_van = layer.self_attn.k_proj(captured['after_input_norm']).cpu()
        V_van = layer.self_attn.v_proj(captured['after_input_norm']).cpu()

    diff_Q = (Q.cpu() - Q_van.cpu()).abs()
    diff_K = (K.cpu() - K_van.cpu()).abs()
    diff_V = (V.cpu() - V_van.cpu()).abs()
    print(f'  {"2. Q projection":35s}  L1={diff_Q.mean():.5f}  Max={diff_Q.max():.5f}')
    print(f'  {"   K projection":35s}  L1={diff_K.mean():.5f}  Max={diff_K.max():.5f}')
    print(f'  {"   V projection":35s}  L1={diff_V.mean():.5f}  Max={diff_V.max():.5f}')

    # ── ZK step 3: attention + o_proj ────────────────────────────────────
    num_heads  = layer.self_attn.num_heads
    head_dim   = layer.self_attn.head_dim
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
    shift = math.sqrt(head_dim) * torch.log(
        (torch.exp(to_float(A, ACCU_LOGSF) / math.sqrt(head_dim)) * ~mask).sum(dim=-1, keepdim=True)
    )
    A -= to_int64(shift, ACCU_LOGSF)
    attn_weights = torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(head_dim)).float() * ~mask
    attn_v = fromto_int64(attn_weights @ V, VALUE_LOGSF)
    save_int(attn_v.transpose(0, 1).contiguous().view(seq_len, embed_dim).float(), 1 << VALUE_LOGSF, 'temp_attn_out.bin')
    os.system(f'./self-attn attn {attn_in_f} {seq_len} {embed_dim} {WORKDIR} {lp} {attn_out_f}')
    os.system('rm -f ./temp_Q.bin ./temp_K.bin ./temp_V.bin ./temp_attn_out.bin')

    attn_v_flat = attn_v.float().transpose(0, 1).contiguous().view(seq_len, embed_dim).cpu()
    with torch.no_grad():
        attn_out_tensor = layer.self_attn.o_proj(attn_v_flat)
    save_int(attn_out_tensor, 1 << SCALING_LOG, attn_out_f)

    diff_attn_v = (attn_v_flat - captured['attn_out_pre_res'].cpu()).abs()
    print(f'  {"3. attn_v (before o_proj)":35s}  L1={diff_attn_v.mean():.5f}  Max={diff_attn_v.max():.5f}')
    diff_o = (attn_out_tensor - captured['attn_out_pre_res'].cpu()).abs()
    print(f'  {"   o_proj output":35s}  L1={diff_o.mean():.5f}  Max={diff_o.max():.5f}')

    # ── ZK step 4: attention skip connection ─────────────────────────────
    os.system(f'./skip-connection {input_file} {attn_out_f} {post_attn_f}')
    cmp('4. after attn skip',
        torch.from_numpy(np.fromfile(post_attn_f, dtype=np.int32).reshape(seq_len, embed_dim)),
        captured['after_attn_skip'])

    # ── ZK step 5: post-attn RMSNorm ─────────────────────────────────────
    X2_int32 = np.fromfile(post_attn_f, dtype=np.int32).reshape(seq_len, embed_dim)
    X2_f64 = torch.tensor(X2_int32, dtype=torch.float64, device=0) / (1 << SCALING_LOG)
    rms_inv2 = 1 / torch.sqrt(torch.mean(X2_f64 ** 2, dim=1) + layer.post_attention_layernorm.variance_epsilon)
    save_int(rms_inv2.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm post_attention {post_attn_f} {seq_len} {embed_dim} {WORKDIR} {lp} {ffn_in_f}')
    os.remove('rms_inv_temp.bin')
    cmp('5. after post-attn RMSNorm',
        torch.from_numpy(np.fromfile(ffn_in_f, dtype=np.int32).reshape(seq_len, embed_dim)),
        captured['after_post_norm'])

    # ── ZK step 6: FFN ────────────────────────────────────────────────────
    os.system(f'./ffn {ffn_in_f} {seq_len} {embed_dim} {hidden_dim} {WORKDIR} {lp} {ffn_out_f}')
    cmp('6. after FFN',
        torch.from_numpy(np.fromfile(ffn_out_f, dtype=np.int32).reshape(seq_len, embed_dim)),
        captured['mlp_out_pre_res'])

    # ── ZK step 7: FFN skip connection ────────────────────────────────────
    os.system(f'./skip-connection {post_attn_f} {ffn_out_f} {output_f}')
    cmp('7. final layer output',
        torch.from_numpy(np.fromfile(output_f, dtype=np.int32).reshape(seq_len, embed_dim)),
        captured['layer_out'])

    for f in [attn_in_f, attn_out_f, post_attn_f, ffn_in_f, ffn_out_f, output_f]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7)
    parser.add_argument('--layer', type=int, default=1, help='Which layer to debug (0-indexed)')
    args = parser.parse_args()

    tokenizer, model, embed_dim, hidden_dim, num_layers, workdir = setup(args.model_size)

    with open(args.input_file) as f:
        text = json.load(f)[0]

    token_ids = tokenizer(text, return_tensors='pt')['input_ids']
    seq_len = token_ids.shape[1]
    print(f'seq_len={seq_len}, embed_dim={embed_dim}, layer={args.layer}')

    # For layers > 0 we need to run the ZK pipeline up to layer-1 to get the right input
    with torch.no_grad():
        embedding = model.model.embed_tokens(token_ids).squeeze(0).float()
    save_int(embedding, 1 << SCALING_LOG, '_dbg_embed.bin')

    from full_run import run_layer
    for i in range(args.layer):
        inp = '_dbg_embed.bin' if i == 0 else f'_dbg_prev_{i-1}.bin'
        out = f'_dbg_prev_{i}.bin'
        run_layer(model.model.layers[i], i, seq_len, embed_dim, hidden_dim, workdir, inp, out)

    input_for_target = '_dbg_embed.bin' if args.layer == 0 else f'_dbg_prev_{args.layer - 1}.bin'

    run_step_by_step(
        model.model.layers[args.layer],
        args.layer,
        seq_len, embed_dim, hidden_dim,
        input_for_target,
        model,
    )

    # cleanup
    for f in ['_dbg_embed.bin'] + [f'_dbg_prev_{i}.bin' for i in range(args.layer)]:
        if os.path.exists(f):
            os.remove(f)
