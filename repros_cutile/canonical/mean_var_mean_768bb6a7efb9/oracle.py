"""cuTile port of mean_var_mean_768bb6a7efb9: ConvNeXtV2 residual + spatial mean + LayerNorm.

Two-kernel pipeline:
1. Spatial-mean kernel: NHWC residual add + 1/81 mean over H*W -> pooled[N, C] bf16.
2. LayerNorm kernel: LN over C dim (correction=0), affine, output bf16[N, C].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 640
H = 9
W = 9
HW = H * W  # 81
EPS = 1.0e-6


@ct.kernel
def _spatial_mean_kernel(
    x0_ptr,      # (N, HW, C) bf16 (via NHWC view)
    x1_ptr,      # (N, HW, C) bf16
    pooled_ptr,  # (N, C) bf16
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)
    x0 = ct.load(
        x0_ptr,
        index=(n, 0, c_block),
        shape=(1, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x1 = ct.load(
        x1_ptr,
        index=(n, 0, c_block),
        shape=(1, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Match torch reference: bf16 add first, then cast to fp32 for accumulation.
    added_bf16 = x0 + x1
    added = ct.astype(added_bf16, ct.float32)
    # Mask padding
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    valid = hw_idx < HW_
    valid_3d = ct.reshape(valid, (1, BLOCK_HW, 1))
    zero_3d = ct.zeros((1, BLOCK_HW, BLOCK_C), dtype=ct.float32)
    masked = ct.where(valid_3d, added, zero_3d)

    summed = ct.sum(masked, axis=1)  # (1, BLOCK_C)
    pooled = summed * (1.0 / HW_)
    ct.store(pooled_ptr, index=(n, c_block), tile=ct.astype(pooled, ct.bfloat16))


@ct.kernel
def _layernorm_kernel(
    pooled_ptr,  # (N, C) bf16
    weight_ptr,  # (C,) bf16
    bias_ptr,    # (C,) bf16
    out_ptr,     # (N, C) bf16
    EPS_: ct.Constant[float],
    N_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    # BLOCK_C must be a power of 2 >= N_C; pad OOB with 0 to mask.
    pooled = ct.astype(
        ct.load(
            pooled_ptr,
            index=(row, 0),
            shape=(1, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    valid = ct.reshape(c_idx < N_C, (1, BLOCK_C))
    zero_2d = ct.zeros((1, BLOCK_C), dtype=ct.float32)
    pooled_masked = ct.where(valid, pooled, zero_2d)

    mean = ct.sum(pooled_masked) * (1.0 / N_C)
    # Centered: keep 0 for invalid positions (so they don't skew variance).
    centered = ct.where(valid, pooled - mean, zero_2d)
    variance = ct.sum(centered * centered) * (1.0 / N_C)
    invstd = ct.rsqrt(variance + EPS_)

    weight = ct.astype(
        ct.load(
            weight_ptr,
            index=(0,),
            shape=(BLOCK_C,),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(
            bias_ptr,
            index=(0,),
            shape=(BLOCK_C,),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    out = centered * invstd * weight_2d + bias_2d
    # Only store the valid slice: use a masked scatter isn't easy; write into a
    # size-1024 output cell but only up to N_C is meaningful — but ct.store writes
    # the whole tile. To avoid OOB writes to the (N, C=640) tensor with tile
    # of shape (1, 1024), we split the store: pass a wider output buffer.
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="10a79381")
def oracle_forward(inputs):
    x0, x1, weight, bias, _s0, _s1, _s2 = inputs
    # NCHW channels-last -> NHWC contiguous zero-copy view.
    x0_flat = x0.permute(0, 2, 3, 1).reshape(N, HW, C)
    x1_flat = x1.permute(0, 2, 3, 1).reshape(N, HW, C)
    pooled = torch.empty_strided(
        (N, C),
        (C, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    BLOCK_HW = 128
    BLOCK_C_SPATIAL = 128
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, C // BLOCK_C_SPATIAL, 1),
        _spatial_mean_kernel,
        (x0_flat, x1_flat, pooled, HW, BLOCK_HW, BLOCK_C_SPATIAL),
    )
    # For LN we need a power-of-2 BLOCK_C >= C=640. Use 1024, but write into a
    # padded scratch buffer to avoid OOB writes.
    BLOCK_C_LN = 1024
    padded_out = torch.zeros((N, BLOCK_C_LN), device=x0.device, dtype=torch.bfloat16)
    padded_in = torch.zeros((N, BLOCK_C_LN), device=x0.device, dtype=torch.bfloat16)
    padded_in[:, :C] = pooled
    padded_weight = torch.zeros((BLOCK_C_LN,), device=x0.device, dtype=torch.bfloat16)
    padded_weight[:C] = weight
    padded_bias = torch.zeros((BLOCK_C_LN,), device=x0.device, dtype=torch.bfloat16)
    padded_bias[:C] = bias

    ct.launch(
        stream,
        (N, 1, 1),
        _layernorm_kernel,
        (padded_in, padded_weight, padded_bias, padded_out, EPS, C, BLOCK_C_LN),
    )
    return padded_out[:, :C].contiguous()
