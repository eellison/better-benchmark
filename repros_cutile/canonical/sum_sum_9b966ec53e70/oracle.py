"""cuTile port of sum_sum_9b966ec53e70: GhostNet BN backward with sliced
channels-last input.

Inputs (arg0 is channels-last bf16 [512, 160, 7, 7], arg1 is channels-last
bf16 [512, 80, 7, 7]):

Outputs:
  clone: NCHW-contiguous bf16 clone of arg0
  copy:  channels-last bf16 (same layout / values as arg0 for the visible copy)
  sum_1: fp32[80] = sum(arg0[:, 80:160, :, :].float(), dim=(0,2,3))
  mul_8: fp32[80] = sum_2 * arg3_1, where sum_2 = sum(slice*sub, dims)
  dense: channels-last bf16 [512, 80, 7, 7] = BN-backward dense output

Kernel strategy:
  1. `_layout_copy_kernel`: read channels-last arg0 and produce two visible
     outputs — NCHW-contiguous `clone_out` and channels-last `copy_out`.
  2. `_partial_reduce_kernel`: per-channel partial `sum(grad)` and
     `sum(grad * centered)` over N*H*W chunks. `grad` is the slice
     `arg0[:, 80:160]` (channels-last); `centered = arg1.float() - mean`.
  3. `_finalize_kernel`: reduce partials, compute sum_1, mul_8, and
     per-channel derived scale factors used by the dense epilogue.
  4. `_epilogue_kernel`: dense channels-last bf16 output combining grad,
     centered, mean_term, correction, and output_scale.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
INPUT_C = 160
C = 80
SLICE_START = 80
H = 7
W = 7
HW = H * W          # 49
K_TOTAL = N * HW    # 25088
OUT_NUMEL = N * C * HW  # 2007040
SCALE = 3.985969387755102e-05


@ct.kernel
def _layout_copy_kernel(
    source_ptr,        # bf16 view of (N, HW, INPUT_C) contig
    clone_out_ptr,     # bf16 view of (N, INPUT_C, HW) contig
    copy_out_ptr,      # bf16 view of (N, HW, INPUT_C) contig
    N_: ct.Constant[int],
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)
    hw_block = ct.bid(2)

    # Load a (1, BLOCK_HW, BLOCK_C) tile from source (N, HW, INPUT_C).
    x = ct.load(
        source_ptr, index=(n, hw_block, c_block),
        shape=(1, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Store to copy_out in-place (same physical layout).
    ct.store(copy_out_ptr, index=(n, hw_block, c_block), tile=x)

    # For clone_out (N, INPUT_C, HW), we need the transposed tile
    # (1, BLOCK_C, BLOCK_HW).
    x_t = ct.permute(x, (0, 2, 1))
    ct.store(clone_out_ptr, index=(n, c_block, hw_block), tile=x_t)


@ct.kernel
def _partial_reduce_kernel(
    src_c_last_ptr,    # bf16 [K, INPUT_C] = view of (N, HW, C)
    act_c_last_ptr,    # bf16 [K, C]      = view of (N, HW, C_out)
    mean_ptr,          # f32  [C]
    partial_sum_ptr,   # f32  [num_k_tiles, C]  = per-tile sum(grad)
    partial_dot_ptr,   # f32  [num_k_tiles, C]  = per-tile sum(grad * centered)
    INPUT_C_: ct.Constant[int],
    C_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    K_TOTAL_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    c_local = ct.arange(BLOCK_C_, dtype=ct.int32)
    k_local = ct.arange(BLOCK_K_, dtype=ct.int32)
    c_idx = c_block * BLOCK_C_ + c_local
    k_idx = k_block * BLOCK_K_ + k_local
    c_valid = ct.reshape(c_idx < C_, (1, BLOCK_C_))
    k_valid = ct.reshape(k_idx < K_TOTAL_, (BLOCK_K_, 1))
    valid = c_valid & k_valid

    # Load grad (slice [K, C_out] from src_c_last) via gather with rank-2 index.
    grad_c_col = SLICE_START_ + c_idx  # shape (BLOCK_C_,)
    grad_c_col_bcast = ct.reshape(grad_c_col, (1, BLOCK_C_))
    grad_c_col_2d = grad_c_col_bcast + ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.int32)
    k_col_bcast = ct.reshape(k_idx, (BLOCK_K_, 1))
    k_col_2d = k_col_bcast + ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.int32)
    # Safe indices for OOB lanes: keep in-bounds so gather doesn't OOB.
    zero_2d = ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.int32)
    k_safe = ct.where(valid, k_col_2d, zero_2d)
    c_safe = ct.where(valid, grad_c_col_2d, zero_2d)
    grad_flat_2d = ct.gather(src_c_last_ptr, (k_safe, c_safe))
    grad_f = ct.astype(grad_flat_2d, ct.float32)
    zero_bk = ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.float32)
    grad_f = ct.where(valid, grad_f, zero_bk)

    # Load activation directly at (k_block, c_block).
    act = ct.load(
        act_c_last_ptr, index=(k_block, c_block),
        shape=(BLOCK_K_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    act_f = ct.astype(act, ct.float32)

    # Load mean[c].
    mean_1d = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C_,),
                      padding_mode=ct.PaddingMode.ZERO)
    mean_f = ct.astype(mean_1d, ct.float32)
    mean_bcast = ct.reshape(mean_f, (1, BLOCK_C_))
    centered = act_f - mean_bcast

    partial_sum = ct.sum(grad_f, axis=0)  # (BLOCK_C,)
    partial_dot = ct.sum(grad_f * centered, axis=0)  # (BLOCK_C,)
    ct.store(partial_sum_ptr, index=(k_block, c_block), tile=ct.reshape(partial_sum, (1, BLOCK_C_)))
    ct.store(partial_dot_ptr, index=(k_block, c_block), tile=ct.reshape(partial_dot, (1, BLOCK_C_)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,     # f32 [num_k_tiles, C]
    partial_dot_ptr,     # f32 [num_k_tiles, C]
    invstd_ptr,          # f32 [C]
    weight_ptr,          # f32 [C]
    sum_out_ptr,         # f32 [C]  == sum_1
    mul8_out_ptr,        # f32 [C]  == sum_2 * weight
    stats_ptr,           # f32 [3, C]  (mean_term, correction_scale, output_scale)
    C_: ct.Constant[int],
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    SCALE_VALUE: ct.Constant[float],
):
    c_block = ct.bid(0)
    c_local = ct.arange(BLOCK_C_, dtype=ct.int32)
    c_idx = c_block * BLOCK_C_ + c_local
    c_valid = c_idx < C_

    tile_local = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tile_valid = tile_local < NUM_TILES
    mask = ct.reshape(tile_valid, (BLOCK_TILES, 1)) & ct.reshape(c_valid, (1, BLOCK_C_))
    zero_2d = ct.zeros((BLOCK_TILES, BLOCK_C_), dtype=ct.float32)
    ps = ct.load(partial_sum_ptr, index=(0, c_block),
                 shape=(BLOCK_TILES, BLOCK_C_),
                 padding_mode=ct.PaddingMode.ZERO)
    pd = ct.load(partial_dot_ptr, index=(0, c_block),
                 shape=(BLOCK_TILES, BLOCK_C_),
                 padding_mode=ct.PaddingMode.ZERO)
    ps_masked = ct.where(mask, ps, zero_2d)
    pd_masked = ct.where(mask, pd, zero_2d)
    sum_value = ct.sum(ps_masked, axis=0)   # (BLOCK_C,)
    dot_value = ct.sum(pd_masked, axis=0)   # (BLOCK_C,)

    invstd = ct.astype(
        ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight = ct.astype(
        ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean_term = sum_value * SCALE_VALUE
    dot_mean = dot_value * SCALE_VALUE
    invstd_sq = invstd * invstd
    correction_scale = dot_mean * invstd_sq
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    # mul_8 = sum_2 * invstd (per eager Repro)
    ct.store(mul8_out_ptr, index=(c_block,), tile=dot_value * invstd)
    ct.store(stats_ptr, index=(0, c_block), tile=ct.reshape(mean_term, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(1, c_block), tile=ct.reshape(correction_scale, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(2, c_block), tile=ct.reshape(output_scale, (1, BLOCK_C_)))


@ct.kernel
def _epilogue_kernel(
    src_c_last_ptr,    # bf16 [K, INPUT_C]  (N*HW, INPUT_C)
    act_c_last_ptr,    # bf16 [K, C]        (N*HW, C)
    mean_ptr,          # f32  [C]
    stats_ptr,         # f32  [3, C]
    dense_out_ptr,     # bf16 [K, C]  (N*HW, C) contiguous
    INPUT_C_: ct.Constant[int],
    C_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    K_TOTAL_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    k_block = ct.bid(0)
    c_block = ct.bid(1)
    c_local = ct.arange(BLOCK_C_, dtype=ct.int32)
    k_local = ct.arange(BLOCK_K_, dtype=ct.int32)
    c_idx = c_block * BLOCK_C_ + c_local
    k_idx = k_block * BLOCK_K_ + k_local
    c_valid = ct.reshape(c_idx < C_, (1, BLOCK_C_))
    k_valid = ct.reshape(k_idx < K_TOTAL_, (BLOCK_K_, 1))
    valid = c_valid & k_valid

    grad_c_col = SLICE_START_ + c_idx  # shape (BLOCK_C_,)
    grad_c_col_bcast = ct.reshape(grad_c_col, (1, BLOCK_C_)) + ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.int32)
    k_col_bcast = ct.reshape(k_idx, (BLOCK_K_, 1)) + ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.int32)
    zero_2d_i = ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.int32)
    k_safe = ct.where(valid, k_col_bcast, zero_2d_i)
    c_safe = ct.where(valid, grad_c_col_bcast, zero_2d_i)
    grad = ct.gather(src_c_last_ptr, (k_safe, c_safe))
    grad_f = ct.astype(grad, ct.float32)

    act = ct.load(
        act_c_last_ptr, index=(k_block, c_block),
        shape=(BLOCK_K_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    act_f = ct.astype(act, ct.float32)

    mean_1d = ct.astype(
        ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean_term_1d = ct.load(
        stats_ptr, index=(0, c_block), shape=(1, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    correction_1d = ct.load(
        stats_ptr, index=(1, c_block), shape=(1, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scale_1d = ct.load(
        stats_ptr, index=(2, c_block), shape=(1, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean_bcast = ct.reshape(mean_1d, (1, BLOCK_C_))
    mean_term_bcast = mean_term_1d
    correction_bcast = correction_1d
    scale_bcast = scale_1d

    centered = act_f - mean_bcast
    correction = centered * correction_bcast
    after_corr = grad_f - correction
    after_mean = after_corr - mean_term_bcast
    result = after_mean * scale_bcast
    out_bf16 = ct.astype(result, ct.bfloat16)
    ct.store(dense_out_ptr, index=(k_block, c_block), tile=out_bf16)


@oracle_impl(hardware="B200", point="3daf9266",
             BLOCK_C=16, BLOCK_K=32, BLOCK_COPY_C=16, BLOCK_COPY_HW=64)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_K, BLOCK_COPY_C, BLOCK_COPY_HW):
    source, activation, mean_bcast, invstd, weight = inputs
    device = source.device
    # Reshape to physical layout for cuTile-friendly indexing.
    # source is channels-last (N, INPUT_C, H, W) with strides (INPUT_C*HW, 1, W*INPUT_C, INPUT_C)
    # Its physical layout is NHWC, i.e. (N, H, W, INPUT_C) contiguous.
    # As a (N, HW, INPUT_C) view: .permute(0,2,3,1).contiguous().view(N, HW, INPUT_C)
    # But since it's already channels-last, we can view it directly.
    source_view = source.permute(0, 2, 3, 1).reshape(N, HW, INPUT_C)
    activation_view = activation.permute(0, 2, 3, 1).reshape(N, HW, C)
    mean_1d = mean_bcast.view(C)

    # Allocate visible outputs.
    clone_out = torch.empty_strided(
        (N, INPUT_C, H, W), (INPUT_C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    copy_out = torch.empty_strided(
        (N, INPUT_C, H, W), (INPUT_C * HW, 1, W * INPUT_C, INPUT_C),
        device=device, dtype=torch.bfloat16,
    )
    # For kernel storage view them by physical layout.
    clone_view = clone_out.view(N, INPUT_C, HW)          # NCHW contiguous
    copy_view = copy_out.permute(0, 2, 3, 1).reshape(N, HW, INPUT_C)  # NHWC contiguous

    stream = torch.cuda.current_stream()
    # 1) Copy visible outputs.
    n_c_tiles = (INPUT_C + BLOCK_COPY_C - 1) // BLOCK_COPY_C
    n_hw_tiles = (HW + BLOCK_COPY_HW - 1) // BLOCK_COPY_HW
    ct.launch(
        stream, (N, n_c_tiles, n_hw_tiles), _layout_copy_kernel,
        (source_view, clone_view, copy_view,
         N, INPUT_C, HW, BLOCK_COPY_C, BLOCK_COPY_HW),
    )

    # 2) Partial reduce.
    src_2d = source_view.view(K_TOTAL, INPUT_C)  # (K, INPUT_C) contiguous
    act_2d = activation_view.view(K_TOTAL, C)    # (K, C) contiguous
    num_k_tiles = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    num_c_tiles = (C + BLOCK_C - 1) // BLOCK_C
    partial_sum = torch.empty((num_k_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, C), device=device, dtype=torch.float32)
    ct.launch(
        stream, (num_c_tiles, num_k_tiles, 1), _partial_reduce_kernel,
        (src_2d, act_2d, mean_1d, partial_sum, partial_dot,
         INPUT_C, C, SLICE_START, K_TOTAL, BLOCK_K, BLOCK_C),
    )

    # 3) Finalize.
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    mul8_out = torch.empty((C,), device=device, dtype=torch.float32)
    stats = torch.empty((3, C), device=device, dtype=torch.float32)
    block_tiles = 1
    while block_tiles < num_k_tiles:
        block_tiles *= 2
    ct.launch(
        stream, (num_c_tiles, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd, weight, sum_out, mul8_out, stats,
         C, num_k_tiles, block_tiles, BLOCK_C, SCALE),
    )

    # 4) Epilogue.
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    dense_view = dense_out.permute(0, 2, 3, 1).reshape(N, HW, C).view(K_TOTAL, C)
    ct.launch(
        stream, (num_k_tiles, num_c_tiles, 1), _epilogue_kernel,
        (src_2d, act_2d, mean_1d, stats, dense_view,
         INPUT_C, C, SLICE_START, K_TOTAL, BLOCK_K, BLOCK_C),
    )

    return clone_out, copy_out, sum_out, mul8_out, dense_out
