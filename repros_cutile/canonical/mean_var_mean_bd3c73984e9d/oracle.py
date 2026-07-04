"""cuTile port of mean_var_mean_bd3c73984e9d: BEiT pooled-token LayerNorm.

Kernel 1 (pooling): for each (batch, channel), sum over tokens t in [1, TOKENS)
  value = residual[b, t, c] + gamma[c] * mlp[b*TOKENS+t, c]  (fp32)
Kernel 2 (LayerNorm per batch): mean/var over hidden -> normalized/affine/rsqrt/HIDDEN.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
LIVE_TOKENS = 196  # 197-1
LIVE_BLOCK = 256   # power-of-2 >= LIVE_TOKENS
HIDDEN = 768
BLOCK_H = 1024


@ct.kernel
def _pooled_sum_kernel(
    mlp_ptr,       # bf16 [BATCH*TOKENS, HIDDEN]
    gamma_ptr,     # f32 [HIDDEN]
    residual_ptr,  # f32 [BATCH, TOKENS, HIDDEN]
    pooled_ptr,    # f32 [BATCH, HIDDEN]
    TOKENS: ct.Constant[int],
    LIVE_TOKENS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    LIVE_BLOCK: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)

    # Load gamma[c]
    gamma_c = ct.load(gamma_ptr, index=(c,), shape=(1,))
    gamma_f = ct.astype(gamma_c, ct.float32)

    # Sum over live tokens t in [1, TOKENS): value = residual[b, t, c] + gamma * mlp[b*TOKENS+t, c]
    # We load tokens 0..LIVE_BLOCK (BLOCK=256, > TOKENS=197) and mask off t=0 and t>=TOKENS.
    tok_idx = ct.arange(LIVE_BLOCK, dtype=ct.int32)
    valid_tok = (tok_idx >= 1) & (tok_idx < TOKENS)

    # residual[b, tok_idx, c] via gather across the (token) axis.
    b_idx = ct.full(shape=(LIVE_BLOCK,), fill_value=b, dtype=ct.int32)
    c_idx = ct.full(shape=(LIVE_BLOCK,), fill_value=c, dtype=ct.int32)
    residual = ct.gather(residual_ptr, (b_idx, tok_idx, c_idx), mask=valid_tok, padding_value=0)

    # mlp[b*TOKENS+tok_idx, c] via gather
    row_idx = ct.full(shape=(LIVE_BLOCK,), fill_value=b * TOKENS, dtype=ct.int32) + tok_idx
    mlp = ct.gather(mlp_ptr, (row_idx, c_idx), mask=valid_tok, padding_value=0)
    mlp_f = ct.astype(mlp, ct.float32)

    zero_f = ct.full(shape=(LIVE_BLOCK,), fill_value=0.0, dtype=ct.float32)
    value = residual + gamma_f * mlp_f
    value_masked = ct.where(valid_tok, value, zero_f)
    total = ct.sum(value_masked)
    ct.store(pooled_ptr, index=(b, c), tile=ct.reshape(total, (1, 1)))


@ct.kernel
def _layernorm_outputs_kernel(
    pooled_ptr,    # f32 [BATCH, HIDDEN]
    weight_ptr,    # f32 [HIDDEN]
    bias_ptr,      # f32 [HIDDEN]
    normed_ptr,    # f32 [BATCH, HIDDEN]
    out_ptr,       # bf16 [BATCH, HIDDEN]
    side_ptr,      # f32 [BATCH]
    HIDDEN: ct.Constant[int],
    LIVE_TOKENS: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    pooled_sum = ct.load(pooled_ptr, index=(row, 0), shape=(1, BLOCK_H),
                         padding_mode=ct.PaddingMode.ZERO)
    pooled_sum = ct.reshape(pooled_sum, (BLOCK_H,))
    pooled = pooled_sum * (1.0 / LIVE_TOKENS)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    valid_1d = cols < HIDDEN
    zero_f = ct.full(shape=(BLOCK_H,), fill_value=0.0, dtype=ct.float32)
    pooled_masked = ct.where(valid_1d, pooled, zero_f)
    mean = ct.sum(pooled_masked) * (1.0 / HIDDEN)
    centered = pooled - mean
    centered_masked = ct.where(valid_1d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-6)
    normed = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    affine = normed * weight + bias

    r_idx = ct.full(shape=(BLOCK_H,), fill_value=row, dtype=ct.int32)
    ct.scatter(normed_ptr, (r_idx, cols), ct.where(valid_1d, normed, zero_f), mask=valid_1d)
    zero_bf = ct.full(shape=(BLOCK_H,), fill_value=0.0, dtype=ct.bfloat16)
    ct.scatter(out_ptr, (r_idx, cols),
               ct.where(valid_1d, ct.astype(affine, ct.bfloat16), zero_bf),
               mask=valid_1d)
    ct.store(side_ptr, index=(row,), tile=ct.reshape(invstd / HIDDEN, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(
    hardware="B200",
    point="f4c82f7a",
    XBLOCK=64,
    RBLOCK=8,
    BLOCK_H=BLOCK_H,
)
def oracle_forward(inputs, *, XBLOCK, RBLOCK, BLOCK_H):
    mlp, gamma, residual, weight, bias, view_shape = inputs
    batch, tokens, hidden = _shape_tuple(view_shape)

    out_shape = (batch, hidden)
    out_stride = _contiguous_stride(out_shape)
    side_shape = (batch, 1)

    pooled = torch.empty_strided(out_shape, out_stride, device=mlp.device, dtype=torch.float32)
    normed = torch.empty_strided(out_shape, out_stride, device=mlp.device, dtype=torch.float32)
    out = torch.empty_strided(out_shape, out_stride, device=mlp.device, dtype=torch.bfloat16)
    side = torch.empty_strided(side_shape, _contiguous_stride(side_shape),
                               device=mlp.device, dtype=torch.float32)
    side_1d = side.view(batch)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, hidden, 1),
        _pooled_sum_kernel,
        (mlp, gamma, residual, pooled, tokens, LIVE_TOKENS, hidden, LIVE_BLOCK),
    )
    ct.launch(
        stream,
        (batch, 1, 1),
        _layernorm_outputs_kernel,
        (pooled, weight, bias, normed, out, side_1d, hidden, LIVE_TOKENS, BLOCK_H),
    )
    return normed, out, side
