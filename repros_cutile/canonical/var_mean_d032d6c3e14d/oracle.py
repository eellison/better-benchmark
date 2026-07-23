"""cuTile port of var_mean_d032d6c3e14d: token-selective residual + LayerNorm.

Only tokens 0 and 1 per batch are normalized. For each selected row (batch, tok in {0,1}),
compute LayerNorm(flat + residual, weight, bias, eps=1e-6) and store into out_base[batch, tok, :].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _selected_token_pair_layernorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN] = [batch*tokens, HIDDEN]
    residual_ptr,  # bf16 [batch, tokens, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    out_ptr,       # bf16 [batch, tokens, HIDDEN]
    HIDDEN: ct.Constant[int],
    TOKENS: ct.Constant[int],
    SELECTED_TOKENS: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    sel = ct.bid(0)
    b = sel // SELECTED_TOKENS
    t = sel - b * SELECTED_TOKENS

    # Load flat[b*TOKENS + t, :] and residual[b, t, :]
    src_row = b * TOKENS + t
    flat = ct.load(flat_ptr, index=(src_row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(b, t, 0), shape=(1, 1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    residual = ct.reshape(residual, (1, BLOCK_H))
    added = ct.astype(flat, ct.float32) + ct.astype(residual, ct.float32)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    valid_1d = cols < HIDDEN
    valid = ct.reshape(valid_1d, (1, BLOCK_H))
    zero_f = ct.full(shape=(1, BLOCK_H), fill_value=0.0, dtype=ct.float32)
    zero_bf = ct.full(shape=(1, BLOCK_H), fill_value=0.0, dtype=ct.bfloat16)

    values = ct.where(valid, added, zero_f)
    mean = ct.sum(values) * (1.0 / HIDDEN)
    centered = added - mean
    centered_masked = ct.where(valid, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    affine = centered * invstd * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_bf = ct.where(valid, affine_bf, zero_bf)

    # Scatter to out_ptr[b, t, cols] with mask
    b_idx = ct.full(shape=(BLOCK_H,), fill_value=b, dtype=ct.int32)
    t_idx = ct.full(shape=(BLOCK_H,), fill_value=t, dtype=ct.int32)
    ct.scatter(out_ptr, (b_idx, t_idx, cols),
               ct.reshape(affine_bf, (BLOCK_H,)),
               mask=valid_1d)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="155170ab", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0 = inputs
    base_shape = _as_shape(shape0)
    batch = int(base_shape[0])
    tokens = int(base_shape[1])
    hidden = int(base_shape[2])
    selected_tokens = 2
    selected_rows = batch * selected_tokens

    out_base = torch.empty_strided(
        base_shape,
        (tokens * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (selected_rows, 1, 1),
        _selected_token_pair_layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, arg3_1, out_base, hidden, tokens, selected_tokens, BLOCK_H),
    )
    return out_base.select(1, 0), out_base.select(1, 1)
