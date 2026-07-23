"""cuTile port of mean_b5152799a946: Visformer residual+BN+spatial mean.

Rewrites channels-last NCHW inputs as contiguous NHWC (zero-copy). For each
(n, c) row the kernel loads all 49 spatial elements, computes residual add,
BatchNorm-affine, bf16 rounding, and 1/49 mean, then writes bf16[N, C].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 768
H = 7
W = 7
HW = H * W  # 49


@ct.kernel
def _visformer_bn_mean_kernel(
    x0_ptr,       # (N, H, W, C) bf16
    x1_ptr,       # (N, H, W, C) bf16
    mean_ptr,     # (C,) bf16
    var_ptr,      # (C,) bf16
    weight_ptr,   # (C,) bf16
    bias_ptr,     # (C,) bf16
    out_ptr,      # (N, C) bf16
    HW_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)
    # Load (1, HW, 1, BLOCK_C) — mean is over the H*W spatial dims.
    # We treat the NHWC tensor viewed as (N, HW, C) with (h, w) flattened.
    x0 = ct.astype(
        ct.load(
            x0_ptr,
            index=(n, 0, c_block),
            shape=(1, BLOCK_HW, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    x1 = ct.astype(
        ct.load(
            x1_ptr,
            index=(n, 0, c_block),
            shape=(1, BLOCK_HW, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    added = ct.astype(x0 + x1, ct.bfloat16)  # bf16 rounding boundary

    mean = ct.astype(ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)

    inv = 1.0 / ct.sqrt(var + 1.0e-5)
    mean_3d = ct.reshape(mean, (1, 1, BLOCK_C))
    inv_3d = ct.reshape(inv, (1, 1, BLOCK_C))
    weight_3d = ct.reshape(weight, (1, 1, BLOCK_C))
    bias_3d = ct.reshape(bias, (1, 1, BLOCK_C))

    normalized = (ct.astype(added, ct.float32) - mean_3d) * inv_3d
    affine = normalized * weight_3d + bias_3d
    rounded = ct.astype(affine, ct.bfloat16)

    # Mask out spatial padding (only first HW are valid).
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    valid = hw_idx < HW_
    valid_3d = ct.reshape(valid, (1, BLOCK_HW, 1))
    zero_3d = ct.zeros((1, BLOCK_HW, BLOCK_C), dtype=ct.float32)
    rounded_f = ct.where(valid_3d, ct.astype(rounded, ct.float32), zero_3d)

    # Spatial mean: sum over axis=1 (HW dim) then / HW.
    summed = ct.sum(rounded_f, axis=1)  # (1, BLOCK_C)
    pooled = summed * (1.0 / HW_)
    # Reshape to (1, BLOCK_C) for a 2D output (N, C) tile at (n, c_block).
    ct.store(out_ptr, index=(n, c_block), tile=ct.astype(pooled, ct.bfloat16))


@oracle_impl(hardware="B200", point="45e1ce96")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _s0, _s1, _s2 = inputs
    # Convert channels-last NCHW -> zero-copy NHWC view, then flatten HW.
    x0_nhwc = arg0_1.permute(0, 2, 3, 1)
    x1_nhwc = arg1_1.permute(0, 2, 3, 1)
    x0_flat = x0_nhwc.reshape(N, HW, C)
    x1_flat = x1_nhwc.reshape(N, HW, C)
    out = torch.empty_strided(
        (N, C),
        (C, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    BLOCK_HW = 64  # power-of-2, HW=49 pads with UNDETERMINED -> mask via bounds check
    # Actually, easier: use BLOCK_HW as power-of-2 next to 49 (64) but need to mask.
    # cuTile: load with padding_mode=ZERO to pad OOB elems.
    # But since we sum over axis=1, we need OOB to be exactly zero. Use ZERO padding.
    BLOCK_C = 128
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, C // BLOCK_C, 1),
        _visformer_bn_mean_kernel,
        (x0_flat, x1_flat, arg2_1, arg3_1, arg4_1, arg5_1, out, HW, C, BLOCK_HW, BLOCK_C),
    )
    return out
