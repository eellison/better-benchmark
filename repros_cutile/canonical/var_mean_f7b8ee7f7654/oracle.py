"""cuTile port of var_mean_f7b8ee7f7654: Blenderbot embedding + LayerNorm.

Uses torch for the embedding gather, then a cuTile row LayerNorm kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _add_and_layernorm_kernel(
    add_ptr,        # bf16 [rows, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    norm_out_ptr,   # bf16 [rows, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf16 = ct.load(add_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = ct.astype(x_bf16, ct.float32)
    mean = ct.sum(x) * (1.0 / HIDDEN_)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    w_f = ct.astype(weight, ct.float32)
    b_f = ct.astype(bias, ct.float32)
    w_2d = ct.reshape(w_f, (1, BLOCK_H))
    b_2d = ct.reshape(b_f, (1, BLOCK_H))
    out = centered * invstd * w_2d + b_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="cd33d4f9")
def oracle_forward(inputs):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    # Use torch for the embedding lookups
    token_emb = token_table.index_select(0, token_ids.view(-1)).view(batch, seq_len, hidden)
    iota_ids = torch.arange(seq_len, device=token_table.device, dtype=torch.int64)
    pos_emb = position_table.index_select(0, iota_ids)  # [seq_len, HIDDEN]

    # mul by 1.0, then add - this preserves bf16 rounding
    # Note: mul by 1.0 in bf16 is identity, but the promotion to fp32 and back matters
    # In Triton kernel it does f32 add then cast to bf16. Let's replicate.
    add_bf16 = torch.empty((batch, seq_len, hidden), device=token_table.device, dtype=torch.bfloat16)
    # add = (token_emb.float() + pos_emb.float().broadcast).to(bf16)
    # We can do this in torch since it uses fp32 add per lane.
    # Actually Triton kernel: adds f32(token) + f32(position) -> bf16 cast
    # In torch: bf16 + bf16 = bf16 (with internal fp32 or not depending on impl).
    # Safer: compute in fp32 and cast down.
    add_bf16 = (token_emb.to(torch.float32) + pos_emb.to(torch.float32)).to(torch.bfloat16)
    # Reshape to (rows, hidden) for cuTile
    add_2d = add_bf16.view(rows, hidden)

    norm_out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    BLOCK_H = hidden  # 2560 - not a power of 2, may not work
    # Actually 2560 = 2^11 * 1.25 - not power of 2
    # Use next power of 2: 4096
    BLOCK_H = 4096
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _add_and_layernorm_kernel,
        (add_2d, weight, bias, norm_out, hidden, BLOCK_H),
    )
    return (
        add_bf16,
        norm_out.view(tuple(int(dim) for dim in shape0)),
        norm_out.view(tuple(int(dim) for dim in shape1)),
        norm_out.view(tuple(int(dim) for dim in shape2)),
    )
