"""cuTile port of mean_var_mean_f541715270de (SCHEDULER_FUSION): BEiT
pool + LayerNorm.

Two cuTile kernels mirror the Triton oracle:
1. Pool kernel: for each (batch, channel) row, compute
   `pooled_sum[batch, channel] = sum_{t=1..196} bf16-round(residual + gamma*mlp)`
   via masked reduction over BLOCK_TOKENS = 256 (pow2 >= 196).
2. LayerNorm kernel: for each batch, divide pooled_sum by LIVE_TOKENS (bf16
   roundtrip), compute row mean/variance, affine, output bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
LIVE_TOKENS = 196
HIDDEN = 768
TOTAL = BATCH * HIDDEN
BLOCK_TOKENS = 256  # pow2 >= LIVE_TOKENS
BLOCK_H = 1024      # pow2 >= HIDDEN


@ct.kernel
def _pool_kernel(
    mlp_ptr,         # bf16 [BATCH*TOKENS*HIDDEN]
    gamma2_ptr,      # bf16 [HIDDEN]
    residual_ptr,    # bf16 [BATCH*TOKENS*HIDDEN]
    pooled_ptr,      # f32  [BATCH*HIDDEN]
    TOTAL_: ct.Constant[int],
    TOKENS_: ct.Constant[int],
    LIVE_TOKENS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    XBLOCK: ct.Constant[int],
    RBLOCK: ct.Constant[int],
):
    x_block = ct.bid(0)
    xidx = x_block * XBLOCK + ct.arange(XBLOCK, dtype=ct.int32)
    xmask = xidx < TOTAL_
    channel = xidx % HIDDEN_
    row = xidx // HIDDEN_

    gamma2 = ct.astype(
        ct.gather(gamma2_ptr, (channel,), mask=xmask),
        ct.float32,
    )
    gamma2_2d = ct.reshape(gamma2, (XBLOCK, 1))
    row_2d = ct.reshape(row, (XBLOCK, 1))
    channel_2d = ct.reshape(channel, (XBLOCK, 1))
    xmask_2d = ct.reshape(xmask, (XBLOCK, 1))

    accum = ct.full((XBLOCK, 1), 0.0, dtype=ct.float32)
    # Loop over token blocks of RBLOCK; LIVE_TOKENS=196, RBLOCK<=256.
    ridx = ct.arange(RBLOCK, dtype=ct.int32)
    ridx_1d2 = ct.reshape(ridx, (1, RBLOCK))
    for roffset in ct.static_iter(range(0, LIVE_TOKENS_, RBLOCK)):
        token = 1 + roffset + ridx_1d2
        mask = xmask_2d & (token < TOKENS_)
        # offsets = (row * TOKENS + token) * HIDDEN + channel
        offsets = (row_2d * TOKENS_ + token) * HIDDEN_ + channel_2d
        mlp = ct.astype(
            ct.gather(mlp_ptr, (offsets,), mask=mask),
            ct.float32,
        )
        residual = ct.astype(
            ct.gather(residual_ptr, (offsets,), mask=mask),
            ct.float32,
        )
        # eager numerics: bf16 arithmetic
        mul_bf = ct.astype(ct.astype(gamma2_2d * mlp, ct.bfloat16), ct.float32)
        value = ct.astype(ct.astype(residual + mul_bf, ct.bfloat16), ct.float32)
        zero_r = ct.full((XBLOCK, RBLOCK), 0.0, dtype=ct.float32)
        contrib = ct.where(mask, value, zero_r)
        # Reduce contribution along token axis, keep XBLOCK dim
        contrib_sum = ct.sum(contrib, axis=1, keepdims=True)
        accum = accum + contrib_sum

    ct.scatter(pooled_ptr, (xidx,), ct.reshape(accum, (XBLOCK,)),
               mask=xmask)


@ct.kernel
def _layernorm_kernel(
    pooled_ptr,     # f32 [BATCH, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    out_ptr,        # bf16 [BATCH, HIDDEN]
    LIVE_TOKENS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    valid = cols < HIDDEN_

    pooled_sum = ct.load(pooled_ptr, index=(row, 0), shape=(1, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
    pooled_f = pooled_sum / LIVE_TOKENS_
    pooled_1d = ct.reshape(pooled_f, (BLOCK_C,))
    # eager bf16 roundtrip on pooled
    pooled = ct.astype(ct.astype(pooled_1d, ct.bfloat16), ct.float32)
    zero_c = ct.full((BLOCK_C,), 0.0, dtype=ct.float32)
    p_masked = ct.where(valid, pooled, zero_c)
    row_mean = ct.sum(p_masked) * (1.0 / HIDDEN_)
    centered = ct.where(valid, pooled - row_mean, zero_c)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    inv_std = ct.rsqrt(variance + 1.0e-6)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32,
    )
    normed = centered * inv_std
    scaled = normed * weight
    out = ct.astype(scaled + bias, ct.bfloat16)
    ct.scatter(
        out_ptr,
        (ct.full((BLOCK_C,), row * HIDDEN_, dtype=ct.int32) + cols,),
        out, mask=valid,
    )


@oracle_impl(hardware="B200", point="e781f0b8")
def oracle_forward(inputs):
    mlp, gamma2, residual, weight, bias, shape = inputs
    device = mlp.device
    viewed = mlp.view(tuple(int(dim) for dim in shape))
    # Flat views for offset-based access
    mlp_flat = viewed.reshape(-1)
    residual_flat = residual.reshape(-1)

    pooled = torch.empty((BATCH * HIDDEN,), device=device, dtype=torch.float32)
    out = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.bfloat16)

    XBLOCK = 64
    RBLOCK = 8
    total = BATCH * HIDDEN

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((total + XBLOCK - 1) // XBLOCK, 1, 1), _pool_kernel,
        (mlp_flat, gamma2, residual_flat, pooled,
         total, TOKENS, LIVE_TOKENS, HIDDEN, XBLOCK, RBLOCK),
    )
    ct.launch(
        stream, (BATCH, 1, 1), _layernorm_kernel,
        (pooled.view(BATCH, HIDDEN), weight, bias, out.view(BATCH * HIDDEN),
         LIVE_TOKENS, HIDDEN, BLOCK_H),
    )
    return out
