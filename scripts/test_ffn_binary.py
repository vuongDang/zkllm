"""
Directly test whether the ./ffn binary is correct by:
1. Computing vanilla ffn_in (from scratch via Python)
2. Running ./ffn on it
3. Comparing ./ffn output vs vanilla FFN(same ffn_in)
4. Also comparing vanilla FFN(ZK ffn_in) vs vanilla FFN(vanilla ffn_in) to see FFN gain

Usage: python test_ffn_binary.py inputs.json --layer 1
"""
import os, json, argparse, math
import torch
import numpy as np
from fileio_utils import save_int, load_int, to_int64, to_float, fromto_int64
from full_run import setup, SCALING_LOG, VALUE_LOGSF, ACCU_LOGSF, rotate_half

_orig = os.system
os.system = lambda cmd: _orig(cmd + ' > /dev/null 2>&1')

WORKDIR = './zkllm-workdir/Llama-2-7b'


def get_vanilla_ffn_in(layer, X_float):
    """Compute vanilla ffn_in (post-attn RMSNorm output) and ffn output via hooks."""
    captured = {}
    def hook_post_norm(m, inp, out):
        captured['ffn_in'] = out.detach().float().squeeze(0)
    def hook_mlp(m, inp, out):
        captured['ffn_out'] = out.detach().float().squeeze(0)
    h1 = layer.post_attention_layernorm.register_forward_hook(hook_post_norm)
    h2 = layer.mlp.register_forward_hook(hook_mlp)
    device = next(layer.parameters()).device
    with torch.no_grad():
        layer(X_float.to(device).unsqueeze(0))
    h1.remove()
    h2.remove()
    return captured['ffn_in'].cpu(), captured['ffn_out'].cpu()


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

    layer = model.model.layers[args.layer]

    # Get ZK pipeline input for this layer
    from full_run import run_layer
    with torch.no_grad():
        embedding = model.model.embed_tokens(token_ids).squeeze(0).float()
    save_int(embedding, 1 << SCALING_LOG, '_tffn_embed.bin')
    for i in range(args.layer):
        inp = '_tffn_embed.bin' if i == 0 else f'_tffn_prev_{i-1}.bin'
        out = f'_tffn_prev_{i}.bin'
        run_layer(model.model.layers[i], i, seq_len, embed_dim, hidden_dim, workdir, inp, out)
    zk_layer_input_f = '_tffn_embed.bin' if args.layer == 0 else f'_tffn_prev_{args.layer-1}.bin'

    # Read ZK layer input as float
    X_int32 = np.fromfile(zk_layer_input_f, dtype=np.int32).reshape(seq_len, embed_dim)
    X_float = torch.tensor(X_int32, dtype=torch.float32) / (1 << SCALING_LOG)

    # ── Part 2: Compute vanilla ffn_in BEFORE ZK pipeline moves rotary_emb to GPU ─
    van_ffn_in, van_ffn_out = get_vanilla_ffn_in(layer, X_float)

    # ── Part 1: Compute ZK ffn_in (from ZK pipeline) ─────────────────────
    lp = f'layer-{args.layer}'
    attn_in_f   = '_tffn_attn_in.bin'
    attn_out_f  = '_tffn_attn_out.bin'
    post_attn_f = '_tffn_post_attn.bin'
    zk_ffn_in_f = '_tffn_zk_ffn_in.bin'
    van_ffn_in_f= '_tffn_van_ffn_in.bin'
    zk_ffn_out_f= '_tffn_zk_ffn_out.bin'

    X_f64 = torch.tensor(X_int32, dtype=torch.float64, device=0) / (1 << SCALING_LOG)
    rms_inv = 1 / torch.sqrt(torch.mean(X_f64**2, dim=1) + layer.input_layernorm.variance_epsilon)
    save_int(rms_inv.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm input {zk_layer_input_f} {seq_len} {embed_dim} {WORKDIR} {lp} {attn_in_f}')
    os.remove('rms_inv_temp.bin')

    os.system(f'./self-attn linear {attn_in_f} {seq_len} {embed_dim} {WORKDIR} {lp} {attn_out_f}')
    Q = load_int('temp_Q.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    K = load_int('temp_K.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    V = load_int('temp_V.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    num_heads = layer.self_attn.num_heads
    head_dim  = layer.self_attn.head_dim
    Q = Q.view(seq_len, num_heads, head_dim).transpose(0,1)
    K = K.view(seq_len, num_heads, head_dim).transpose(0,1)
    V = V.view(seq_len, num_heads, head_dim).transpose(0,1)
    layer.self_attn.rotary_emb.to(0)
    cos, sin = layer.self_attn.rotary_emb(Q.float(), seq_len=seq_len)
    cos = cos.squeeze(0).squeeze(0)
    sin = sin.squeeze(0).squeeze(0)
    Q = (Q * cos.unsqueeze(0) + rotate_half(Q) * sin.unsqueeze(0)).to(torch.float64)
    K = (K * cos.unsqueeze(0) + rotate_half(K) * sin.unsqueeze(0)).to(torch.float64)
    A = to_int64(Q @ K.transpose(-2,-1), ACCU_LOGSF)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=0, dtype=bool), diagonal=1)
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    shift = math.sqrt(head_dim) * torch.log(
        (torch.exp(to_float(A, ACCU_LOGSF) / math.sqrt(head_dim)) * ~mask).sum(dim=-1, keepdim=True))
    A -= to_int64(shift, ACCU_LOGSF)
    attn_weights = torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(head_dim)).float() * ~mask
    attn_v = fromto_int64(attn_weights @ V, VALUE_LOGSF)
    save_int(attn_v.transpose(0,1).contiguous().view(seq_len, embed_dim).float(), 1<<VALUE_LOGSF, 'temp_attn_out.bin')
    os.system(f'./self-attn attn {attn_in_f} {seq_len} {embed_dim} {WORKDIR} {lp} {attn_out_f}')
    os.system('rm -f ./temp_Q.bin ./temp_K.bin ./temp_V.bin ./temp_attn_out.bin')
    attn_v_flat = attn_v.float().transpose(0,1).contiguous().view(seq_len, embed_dim).cpu()
    with torch.no_grad():
        attn_out_tensor = layer.self_attn.o_proj(attn_v_flat)
    save_int(attn_out_tensor, 1 << SCALING_LOG, attn_out_f)
    os.system(f'./skip-connection {zk_layer_input_f} {attn_out_f} {post_attn_f}')

    X2_int32 = np.fromfile(post_attn_f, dtype=np.int32).reshape(seq_len, embed_dim)
    X2_f64 = torch.tensor(X2_int32, dtype=torch.float64, device=0) / (1 << SCALING_LOG)
    rms_inv2 = 1 / torch.sqrt(torch.mean(X2_f64**2, dim=1) + layer.post_attention_layernorm.variance_epsilon)
    save_int(rms_inv2.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm post_attention {post_attn_f} {seq_len} {embed_dim} {WORKDIR} {lp} {zk_ffn_in_f}')
    os.remove('rms_inv_temp.bin')

    save_int(van_ffn_in, 1 << SCALING_LOG, van_ffn_in_f)

    # ── Part 3: Run ./ffn on ZK ffn_in ────────────────────────────────────
    os.system(f'./ffn {zk_ffn_in_f} {seq_len} {embed_dim} {hidden_dim} {WORKDIR} {lp} {zk_ffn_out_f}')

    # ── Part 4: Compute vanilla FFN on ZK ffn_in and vanilla ffn_in ───────
    zk_ffn_in_float = torch.tensor(
        np.fromfile(zk_ffn_in_f, dtype=np.int32).reshape(seq_len, embed_dim), dtype=torch.float32
    ) / (1 << SCALING_LOG)

    device = next(layer.parameters()).device
    with torch.no_grad():
        van_ffn_of_zk_in = layer.mlp(zk_ffn_in_float.to(device)).cpu()
    van_ffn_of_van_in = van_ffn_out

    zk_ffn_out_int = torch.tensor(
        np.fromfile(zk_ffn_out_f, dtype=np.int32).reshape(seq_len, embed_dim), dtype=torch.float32
    ) / (1 << SCALING_LOG)

    print(f'\n=== FFN binary test (layer {args.layer}) ===')

    # Test A: does ZK binary FFN match vanilla FFN given same (ZK) input?
    diff_A = (zk_ffn_out_int - van_ffn_of_zk_in.cpu()).abs()
    print(f'  ZK binary FFN(ZK ffn_in) vs vanilla FFN(ZK ffn_in):')
    print(f'    L1={diff_A.mean():.5f}  Max={diff_A.max():.5f}')

    # Test B: FFN gain — how much does 0.975 input error amplify?
    diff_ffn_in = (zk_ffn_in_float - van_ffn_in.cpu()).abs()
    diff_B = (van_ffn_of_zk_in - van_ffn_of_van_in.cpu()).abs()
    print(f'  ffn_in error:  L1={diff_ffn_in.mean():.5f}  Max={diff_ffn_in.max():.5f}')
    print(f'  vanilla FFN(ZK ffn_in) vs vanilla FFN(van ffn_in) [FFN gain test]:')
    print(f'    L1={diff_B.mean():.5f}  Max={diff_B.max():.5f}')

    print(f'\n  Magnitudes: van FFN(van) max={van_ffn_of_van_in.abs().max():.3f}')
    print(f'              ZK binary FFN max={zk_ffn_out_int.abs().max():.3f}')

    for f in [attn_in_f, attn_out_f, post_attn_f, zk_ffn_in_f, van_ffn_in_f, zk_ffn_out_f,
              '_tffn_embed.bin'] + [f'_tffn_prev_{i}.bin' for i in range(args.layer)]:
        if os.path.exists(f): os.remove(f)
