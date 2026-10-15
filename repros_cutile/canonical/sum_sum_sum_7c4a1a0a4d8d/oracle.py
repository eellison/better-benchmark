"""cuTile port of sum_sum_sum_7c4a1a0a4d8d: ConvNeXtV2 GRN backward + avg-pool backward.

Two-kernel plan:
  1. `_row_kernel`: (N,) blocks. Per-row, compute the GRN/LN-backward algebra
     (weighted grad, normalized centered, row-local weighted_sum and dot_sum,
     and the `base = invstd/C * (weighted*C - weighted_sum - normalized*dot)`).
     Stores `base_bf16[n, :]` and atomic-adds `grad*normalized` into `out0[c]`
     and `grad` into `out1[c]`.
  2. `_expand_kernel`: (N, cdiv(HW, BLOCK_HW)) blocks. Per-(n, hw-tile), load
     `base_bf16[n, :]`, compute `out = (base.f32 * INV_HW).bf16`, store to
     `out2[n, :, h, w]` at every (h, w) position in the tile (that's the
     expand), and atomic-add `out.f32 * HW` to `out3[c]` (only from the
     hw==0 tile so we don't over-count).

The bf16 rounding for out2 = base * INV_HW matches Triton's
`.to(bfloat16, fp_downcast_rounding='rtne')` since cuTile's default cast is
round-to-nearest-even.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 640
HW = 49
H = 7
W = 7
INV_C = 1.0 / 640.0
INV_HW = 1.0 / 49.0


@ct.kernel
def _row_kernel(
    grad_ptr,      # bf16 [N, C]
    weight_ptr,    # f32  [C]
    x_ptr,         # bf16 [N, C] (nhwc squeezed)
    mean_ptr,      # f32  [N]
    invstd_ptr,   # f32  [N]
    base_bf_ptr,   # bf16 [N, C]  output
    out0_ptr,      # f32  [C]  atomic accumulator
    out1_ptr,      # f32  [C]  atomic accumulator
    C_: ct.Constant[int],
    INV_C_: ct.Constant[float],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    grad_bf = ct.load(grad_ptr, index=(n, 0), shape=(1, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    grad = ct.astype(grad_bf, ct.float32)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    x_bf = ct.load(x_ptr, index=(n, 0), shape=(1, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)
    mean_v = ct.load(mean_ptr, index=(n,), shape=(1,))
    invstd_v = ct.load(invstd_ptr, index=(n,), shape=(1,))
    mean_2d = ct.reshape(mean_v, (1, 1))
    invstd_2d = ct.reshape(invstd_v, (1, 1))

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = ct.reshape(cols < C_, (1, BLOCK_C))
    zero_f = ct.astype(0.0, ct.float32)

    weighted = grad * weight_2d
    centered = x - mean_2d
    normalized = centered * invstd_2d

    weighted_masked = ct.where(col_mask, weighted, zero_f)
    dot_masked = ct.where(col_mask, weighted * normalized, zero_f)
    weighted_sum = ct.sum(weighted_masked, axis=1)  # (1,)
    dot_sum = ct.sum(dot_masked, axis=1)             # (1,)
    weighted_sum_2d = ct.reshape(weighted_sum, (1, 1))
    dot_sum_2d = ct.reshape(dot_sum, (1, 1))

    term = weighted * C_ - weighted_sum_2d
    term = term - normalized * dot_sum_2d
    base = (invstd_2d * INV_C_) * term
    base_bf = ct.astype(base, ct.bfloat16)
    ct.store(base_bf_ptr, index=(n, 0), tile=base_bf)

    # Atomic reductions: cols index into [C]. Mask invalid cols to OOB.
    invalid_col = ct.full((BLOCK_C,), C_, dtype=ct.int32)
    col_safe = ct.where(cols < C_, cols, invalid_col)
    # Use grad_f32 * normalized_1d (reshape to 1D).
    grad_1d = ct.reshape(grad, (BLOCK_C,))
    normalized_1d = ct.reshape(normalized, (BLOCK_C,))
    ct.atomic_add(out0_ptr, (col_safe,), grad_1d * normalized_1d)
    ct.atomic_add(out1_ptr, (col_safe,), grad_1d)


@ct.kernel
def _expand_kernel(
    base_bf_ptr,   # bf16 [N, C]
    out2_ptr,      # bf16 [N, C, H, W]
    out3_ptr,      # f32  [C] atomic accumulator
    C_: ct.Constant[int],
    H_: ct.Constant[int],
    W_: ct.Constant[int],
    HW_: ct.Constant[int],
    INV_HW_: ct.Constant[float],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    base_bf = ct.load(base_bf_ptr, index=(n, 0), shape=(1, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    base_f = ct.astype(base_bf, ct.float32)
    out_f = base_f * INV_HW_
    out_bf = ct.astype(out_f, ct.bfloat16)   # (1, BLOCK_C)

    # Store to all (h, w) positions in out2[n, :, h, w]. out2 shape is
    # [N, C, H, W] contiguous, so out2[n, c, h, w] = out2_flat[n*C*H*W + c*H*W + h*W + w].
    # We iterate over h*W = 7*7 = 49 positions.
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_active = cols < C_

    for h in range(H_):
        for w in range(W_):
            # out2 has shape (N, C, H, W). Use scatter with 4D indices.
            n_t = ct.full((1, BLOCK_C), n, dtype=ct.int32)
            c_t = ct.reshape(cols, (1, BLOCK_C))
            h_t = ct.full((1, BLOCK_C), h, dtype=ct.int32)
            w_t = ct.full((1, BLOCK_C), w, dtype=ct.int32)
            mask2d = ct.reshape(col_active, (1, BLOCK_C))
            ct.scatter(out2_ptr, (n_t, c_t, h_t, w_t), out_bf, mask=mask2d)

    # atomic add to out3: sum over n of out_bf.f32 * HW, once per (n, c).
    invalid_col = ct.full((BLOCK_C,), C_, dtype=ct.int32)
    col_safe = ct.where(cols < C_, cols, invalid_col)
    out_bf_1d = ct.reshape(out_bf, (BLOCK_C,))
    out_f32_scaled = ct.astype(out_bf_1d, ct.float32) * HW_
    ct.atomic_add(out3_ptr, (col_safe,), out_f32_scaled)


@oracle_impl(
    hardware="B200",
    point="fd237320",
    ROW_BLOCK_C=1024,
    OUT_BLOCK=64,
    REDUCE_BLOCK_C=8,
    num_warps=8,
)
def oracle_forward(inputs, *, ROW_BLOCK_C, OUT_BLOCK, REDUCE_BLOCK_C, num_warps):
    del OUT_BLOCK, REDUCE_BLOCK_C, num_warps
    (
        grad_flat, weight, x_nhwc, mean, invstd,
        _s0, _s1, _s2, _s3, _s4, _s5, out_shape_param,
    ) = inputs
    device = grad_flat.device
    out_shape = tuple(int(d) for d in out_shape_param)  # (128, 640, 7, 7)

    # Views
    grad_2d = grad_flat  # (N, C) already
    x_2d = x_nhwc.view(N, C)
    mean_1d = mean.view(N)
    invstd_1d = invstd.view(N)

    base_bf = torch.empty_strided((N, C), (C, 1), device=device, dtype=torch.bfloat16)
    out0 = torch.zeros((C,), device=device, dtype=torch.float32)
    out1 = torch.zeros((C,), device=device, dtype=torch.float32)
    out2 = torch.zeros(out_shape, device=device, dtype=torch.bfloat16)
    out3 = torch.zeros((C,), device=device, dtype=torch.float32)

    BLOCK_C = ROW_BLOCK_C  # 1024, power of 2, covers C=640

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N, 1, 1), _row_kernel,
        (grad_2d, weight, x_2d, mean_1d, invstd_1d,
         base_bf, out0, out1,
         C, INV_C, BLOCK_C),
    )
    ct.launch(
        stream, (N, 1, 1), _expand_kernel,
        (base_bf, out2, out3,
         C, H, W, HW, INV_HW, BLOCK_C),
    )

    return out0, out1, out2, out3
