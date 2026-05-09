"""
Test where the 0.912 attention error comes from:
1. Use VANILLA Q, K, V in the Python simulation → should give ~0 error
2. Use ZK Q, K, V in the Python simulation → the 0.912 we see
3. Use vanilla input to identify if Q/K/V quantization is the source

Usage: python test_attn_error.py inputs.json --layer 1
"""
import os, json, argparse, math
import torch
import numpy as np
from fileio_utils import save_int, load_int, to_int64, to_float, fromto_int64
from full_run import setup, SCALING_LOG, VALUE_LOGSF, ACCU_LOGSF, rotate_half

_orig = os.system
os.system = lambda cmd: _orig(cmd + ' > /dev/null 2>&1')

WORKDIR = './zkllm-workdir/Llama-2-7b'


def run_attn_sim(Q_float, K_float, V_float, layer, seq_len):
    """Run the Python attention simulation and o_proj."""
    num_heads = layer.self_attn.num_heads
    head_dim  = layer.self_attn.head_dim
    Q = torch.tensor(Q_float, dtype=torch.float32).view(seq_len, num_heads, head_dim).transpose(0, 1)
    K = torch.tensor(K_float, dtype=torch.float32).view(seq_len, num_heads, head_dim).transpose(0, 1)
    V = torch.tensor(V_float, dtype=torch.float32).view(seq_len, num_heads, head_dim).transpose(0, 1)

    layer.self_attn.rotary_emb.to('cpu')
    cos, sin = layer.self_attn.rotary_emb(Q.float().to('cpu'), seq_len=seq_len)
    cos = cos.squeeze(0).squeeze(0)
    sin = sin.squeeze(0).squeeze(0)
    Q = (Q * cos.unsqueeze(0) + rotate_half(Q) * sin.unsqueeze(0)).to(torch.float64)
    K = (K * cos.unsqueeze(0) + rotate_half(K) * sin.unsqueeze(0)).to(torch.float64)

    A = to_int64(Q @ K.transpose(-2, -1), ACCU_LOGSF)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=bool), diagonal=1)
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    shift = math.sqrt(head_dim) * torch.log(
        (torch.exp(to_float(A, ACCU_LOGSF) / math.sqrt(head_dim)) * ~mask).sum(dim=-1, keepdim=True))
    A -= to_int64(shift, ACCU_LOGSF)
    attn_weights = torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(head_dim)).float() * ~mask
    attn_v = fromto_int64(attn_weights @ V, VALUE_LOGSF)

    attn_v_flat = attn_v.float().transpose(0, 1).contiguous().view(seq_len, -1).cpu()
    with torch.no_grad():
        attn_out = layer.self_attn.o_proj(attn_v_flat)
    return attn_out


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
    layer = model.model.layers[args.layer]

    # Get the ZK pipeline input for this layer
    from full_run import run_layer
    with torch.no_grad():
        embedding = model.model.embed_tokens(token_ids).squeeze(0).float()
    save_int(embedding, 1 << SCALING_LOG, '_tattn_embed.bin')
    for i in range(args.layer):
        inp = '_tattn_embed.bin' if i == 0 else f'_tattn_prev_{i-1}.bin'
        out = f'_tattn_prev_{i}.bin'
        run_layer(model.model.layers[i], i, seq_len, embed_dim, hidden_dim, workdir, inp, out)
    zk_layer_input_f = '_tattn_embed.bin' if args.layer == 0 else f'_tattn_prev_{args.layer-1}.bin'

    X_int32 = np.fromfile(zk_layer_input_f, dtype=np.int32).reshape(seq_len, embed_dim)
    X_float = torch.tensor(X_int32, dtype=torch.float32) / (1 << SCALING_LOG)

    # ── Capture vanilla attention output ─────────────────────────────────
    captured = {}
    def hook_attn(m, inp, out): captured['attn_out'] = out[0].detach().float().squeeze(0).cpu()
    def hook_qkv(m, inp, out): captured['attn_in'] = out.detach().float().squeeze(0).cpu()
    h1 = layer.self_attn.register_forward_hook(hook_attn)
    h2 = layer.input_layernorm.register_forward_hook(hook_qkv)
    with torch.no_grad():
        layer(X_float.unsqueeze(0))
    h1.remove(); h2.remove()
    van_attn_out = captured['attn_out']
    van_attn_in  = captured['attn_in']

    # ── Get ZK binary Q, K, V ─────────────────────────────────────────────
    lp = f'layer-{args.layer}'
    attn_in_f = '_tattn_attn_in.bin'
    attn_out_f = '_tattn_attn_out.bin'
    X_f64 = torch.tensor(X_int32, dtype=torch.float64) / (1 << SCALING_LOG)
    rms_inv = 1 / torch.sqrt(torch.mean(X_f64**2, dim=1) + layer.input_layernorm.variance_epsilon)
    save_int(rms_inv.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm input {zk_layer_input_f} {seq_len} {embed_dim} {WORKDIR} {lp} {attn_in_f}')
    os.remove('rms_inv_temp.bin')
    os.system(f'./self-attn linear {attn_in_f} {seq_len} {embed_dim} {WORKDIR} {lp} {attn_out_f}')
    Q_zk = load_int('temp_Q.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    K_zk = load_int('temp_K.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    V_zk = load_int('temp_V.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    os.system('rm -f temp_Q.bin temp_K.bin temp_V.bin')

    # ── Compute vanilla Q, K, V from vanilla attn_in ─────────────────────
    with torch.no_grad():
        Q_van = layer.self_attn.q_proj(van_attn_in).numpy()
        K_van = layer.self_attn.k_proj(van_attn_in).numpy()
        V_van = layer.self_attn.v_proj(van_attn_in).numpy()

    print(f'\n=== Attention error analysis (layer {args.layer}) ===')
    print(f'  Q_van max={np.abs(Q_van).max():.3f}, Q_zk max={Q_zk.abs().max():.3f}')
    print(f'  K_van max={np.abs(K_van).max():.3f}, K_zk max={K_zk.abs().max():.3f}')
    print(f'  V_van max={np.abs(V_van).max():.3f}, V_zk max={V_zk.abs().max():.3f}')

    # Test A: Python sim with VANILLA Q, K, V → should be ~0 error (pure sim bias)
    attn_out_van_qkv = run_attn_sim(Q_van, K_van, V_van, layer, seq_len)
    err_A = (attn_out_van_qkv - van_attn_out).abs()
    print(f'\n  [A] Python sim(vanilla Q,K,V) vs vanilla attn: L1={err_A.mean():.5f}  Max={err_A.max():.5f}')
    print(f'      (error from int64 quantization + sim vs PyTorch attention)')

    # Test B: Python sim with ZK Q, K, V → same as what debug_layer.py computes
    attn_out_zk_qkv = run_attn_sim(Q_zk.cpu().numpy(), K_zk.cpu().numpy(), V_zk.cpu().numpy(), layer, seq_len)
    err_B = (attn_out_zk_qkv - van_attn_out).abs()
    print(f'\n  [B] Python sim(ZK Q,K,V) vs vanilla attn:     L1={err_B.mean():.5f}  Max={err_B.max():.5f}')
    print(f'      (error from Q/K/V quantization + int64)')

    err_BA = (attn_out_zk_qkv - attn_out_van_qkv).abs()
    print(f'\n  [B-A] Effect of Q/K/V quantization alone:     L1={err_BA.mean():.5f}  Max={err_BA.max():.5f}')

    for f in ['_tattn_embed.bin', attn_in_f, attn_out_f] + \
             [f'_tattn_prev_{i}.bin' for i in range(args.layer)]:
        if os.path.exists(f): os.remove(f)
