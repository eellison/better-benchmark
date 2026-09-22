"""cuTile port of sum_sum_b1354c2a333b: ShuffleNet channel-shuffle + BN-backward.

Reproduces Repro.forward:
  Channel-shuffle (returns view_1 bf16 [128, 464, 7, 7] contiguous):
      slice_1 = arg0[:, 0:232, :, :]
      cat = [slice_1, arg1] on dim 1
      view = reshape([128, 232, 2, 7, 7])
      permute(0, 2, 1, 3, 4) + clone -> [128, 2, 232, 7, 7]
      view_1 = reshape([128, 464, 7, 7])

  BN backward:
      affine = ((arg2.f32 - arg3) * arg4) * arg5 + arg6                (bf16)
      relu   = relu(affine)                                              (bf16)
      le     = relu <= 0
      slice_2 = view_1[:, 232:464]                                       (bf16)
      where  = torch.where(le, arg7_scalar, slice_2)                     (bf16)
      sum_1  = sum(where.f32, [0, 2, 3])                                 (f32[C])
      sum_2  = sum(where.f32 * (arg2.f32 - arg3), [0, 2, 3])             (f32[C])
      mul_10 = sum_2 * arg4                                              (scale-grad vec)
      mean_term = sum_1 * SCALE
      coeff = (sum_2 * SCALE) * (arg4^2)
      out = ((where.f32 - centered * coeff) - mean_term) * (arg4 * arg5) (f32)
  return (view_1, sum_1, mul_10, out.bf16)

Four cuTile kernels; ALL reductions kept inside `@ct.kernel` bodies to
match Triton fairness (Triton port uses 3: shuffle+partial-reduce fused,
finalize, epilogue):
1. `_shuffle_kernel`: materializes `view_1` bf16 contiguous.
2. `_producer_partial_kernel`: computes bf16 `where` producer in
   channels-last flat memory AND per-tile per-channel partial reductions
   for sum_1 and sum_2 (in-kernel `ct.sum` over BLOCK_R rows).
3. `_finalize_kernel`: reduces partials over the tile axis with
   `ct.sum(..., axis=0)` and writes sum_1, sum_2, mul_10.
4. `_epilogue_nhwc_kernel`: computes BN-backward grad output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 232
OUT_C = 464
H = 7
W = 7
HW = H * W
SCALE = 0.00015943877551020407


def _largest_pow2_divisor(n: int) -> int:
    return n & -n


@ct.kernel
def _shuffle_kernel(
    arg0_ptr,    # bf16 flat [N*OUT_C*HW] (arg0 contiguous NCHW)
    arg1_ptr,    # bf16 flat [N*C*HW]    (arg1 channels-last NHWC memory)
    out_ptr,     # bf16 flat [N*OUT_C*HW] (view_1 contiguous NCHW)
    C: ct.Constant[int],
    OUT_C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    # out is contiguous NCHW: idx -> (n, new_c, hw)
    hw = offsets % HW
    tmp = offsets // HW
    new_c = tmp % OUT_C
    n = tmp // OUT_C

    which_half = new_c // C  # 0 or 1
    c = new_c - which_half * C
    old_c = c * 2 + which_half  # channel in cat[0..OUT_C)

    from_arg0 = old_c < C  # if old_c in [0, C) -> arg0; else arg1

    # arg0 is contiguous NCHW: n*OUT_C*HW + old_c*HW + hw
    arg0_off = n * OUT_C * HW + old_c * HW + hw

    # arg1 is channels-last NHWC in memory: c fastest. n*C*HW + hw*C + arg1_c
    arg1_c = old_c - C  # only valid when old_c >= C
    arg1_off = n * C * HW + hw * C + arg1_c

    a0 = ct.gather(arg0_ptr, arg0_off)
    a1 = ct.gather(arg1_ptr, arg1_off)
    out = ct.where(from_arg0, a0, a1)
    ct.store(out_ptr, index=(pid,), tile=out)


@ct.kernel
def _producer_partial_kernel(
    slice2_ptr,      # bf16 (N, C, HW) contiguous
    arg2_ptr,        # bf16 (N, HW, C) channels-last flat
    fill_ptr,        # bf16 [1]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    producer_ptr,    # bf16 (N, HW, C) channels-last flat
    partial_sum_ptr, # f32 (NUM_TILES, C) partial sum(where_val)
    partial_dot_ptr, # f32 (NUM_TILES, C) partial sum(where_val * centered)
    C_C: ct.Constant[int],
    NHW_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Per-tile per-channel-block reduction over N*HW.

    Grid: (num_tiles, num_c_blocks). Each program reduces BLOCK_R rows of
    N*HW for BLOCK_C channels, writing:
      * producer[n, hw, c] = where_out (bf16, NHWC-flat)
      * partial_sum[tile, c] = sum over BLOCK_R of where_val
      * partial_dot[tile, c] = sum over BLOCK_R of where_val * centered
    """
    tile = ct.bid(0)
    c_block = ct.bid(1)

    row_arange = ct.arange(BLOCK_R, dtype=ct.int32)
    rows = tile * BLOCK_R + row_arange
    row_valid = rows < NHW_C

    col_arange = ct.arange(BLOCK_C, dtype=ct.int32)
    cols = c_block * BLOCK_C + col_arange
    col_valid = cols < C_C

    n = rows // HW_C
    hw = rows - n * HW_C

    # Compute (BLOCK_R, BLOCK_C) offsets
    n_2d = ct.reshape(n, (BLOCK_R, 1))
    hw_2d = ct.reshape(hw, (BLOCK_R, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_C))
    zero_i = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)

    # arg2/producer are NHWC-flat: offset = n * HW * C + hw * C + c
    nhwc_off = (n_2d + zero_i) * (HW_C * C_C) + (hw_2d + zero_i) * C_C + (cols_2d + zero_i)
    # slice2 is NCHW-flat: offset = n * C * HW + c * HW + hw
    slice2_off = (n_2d + zero_i) * (C_C * HW_C) + (cols_2d + zero_i) * HW_C + (hw_2d + zero_i)

    row_valid_2d = ct.reshape(row_valid, (BLOCK_R, 1))
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_C))
    valid = row_valid_2d & col_valid_2d

    arg2_bf = ct.gather(arg2_ptr, nhwc_off)
    slice2_bf = ct.gather(slice2_ptr, slice2_off)
    mean = ct.gather(mean_ptr, cols_2d + zero_i)
    invstd = ct.gather(invstd_ptr, cols_2d + zero_i)
    weight = ct.gather(weight_ptr, cols_2d + zero_i)
    bias = ct.gather(bias_ptr, cols_2d + zero_i)

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill, ct.float32)

    arg2_f = ct.astype(arg2_bf, ct.float32)
    centered = arg2_f - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)

    zero_bf = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.bfloat16)
    relu_bf = ct.where(affine_bf > zero_bf, affine_bf, zero_bf)
    le = relu_bf <= zero_bf

    ones_bc = ct.full((BLOCK_R, BLOCK_C), 1.0, dtype=ct.float32)
    fill_bc_bf = ct.astype(ones_bc * fill_f, ct.bfloat16)
    where_out = ct.where(le, fill_bc_bf, slice2_bf)

    ct.scatter(producer_ptr, nhwc_off, where_out, mask=valid)

    where_f = ct.astype(where_out, ct.float32)
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    where_masked = ct.where(valid, where_f, zero_f)
    centered_masked = ct.where(valid, centered, zero_f)
    p_sum = ct.sum(where_masked, axis=0)  # (BLOCK_C,)
    p_dot = ct.sum(where_masked * centered_masked, axis=0)

    partial_off = tile * C_C + cols
    ct.scatter(partial_sum_ptr, partial_off, p_sum, mask=col_valid)
    ct.scatter(partial_dot_ptr, partial_off, p_dot, mask=col_valid)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 (NUM_TILES, C)
    partial_dot_ptr,   # f32 (NUM_TILES, C)
    invstd_ptr,        # f32 [C]
    sum1_ptr,          # f32 [C]
    sum2_ptr,          # f32 [C]
    mul10_ptr,         # f32 [C]
    C_C: ct.Constant[int],
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Reduce partials over the tile axis using ct.sum (axis=0)."""
    c_block = ct.bid(0)
    tiles = ct.arange(BLOCK_TILES, dtype=ct.int32)
    cols_local = ct.arange(BLOCK_C, dtype=ct.int32)
    cols = c_block * BLOCK_C + cols_local
    col_valid = cols < C_C
    tile_valid = tiles < NUM_TILES

    tiles_2d = ct.reshape(tiles, (BLOCK_TILES, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_C))
    zero_i = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.int32)
    offsets = (tiles_2d + zero_i) * C_C + (cols_2d + zero_i)
    tile_valid_2d = ct.reshape(tile_valid, (BLOCK_TILES, 1))
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_C))
    valid = tile_valid_2d & col_valid_2d
    zero_f = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)

    p_sum = ct.gather(partial_sum_ptr, offsets)
    p_dot = ct.gather(partial_dot_ptr, offsets)
    p_sum_m = ct.where(valid, p_sum, zero_f)
    p_dot_m = ct.where(valid, p_dot, zero_f)

    s1 = ct.sum(p_sum_m, axis=0)  # (BLOCK_C,)
    s2 = ct.sum(p_dot_m, axis=0)
    invstd = ct.gather(invstd_ptr, cols)

    ct.scatter(sum1_ptr, cols, s1, mask=col_valid)
    ct.scatter(sum2_ptr, cols, s2, mask=col_valid)
    ct.scatter(mul10_ptr, cols, s2 * invstd, mask=col_valid)


@ct.kernel
def _epilogue_nhwc_kernel(
    producer_ptr,    # bf16 flat NHWC [total]
    arg2_ptr,        # bf16 flat NHWC [total]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    sum1_ptr,        # f32 [C]
    sum2_ptr,        # f32 [C]
    out_ptr,         # bf16 flat NHWC [total]
    C: ct.Constant[int],
    SCALE_VALUE: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    producer = ct.load(producer_ptr, index=(pid,), shape=(BLOCK,))
    arg2 = ct.load(arg2_ptr, index=(pid,), shape=(BLOCK,))

    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = offsets % C

    mean = ct.gather(mean_ptr, c_idx)
    invstd = ct.gather(invstd_ptr, c_idx)
    weight = ct.gather(weight_ptr, c_idx)
    sum1 = ct.gather(sum1_ptr, c_idx)
    sum2 = ct.gather(sum2_ptr, c_idx)

    producer_f = ct.astype(producer, ct.float32)
    arg2_f = ct.astype(arg2, ct.float32)
    centered = arg2_f - mean

    mean_term = sum1 * SCALE_VALUE
    coeff = (sum2 * SCALE_VALUE) * (invstd * invstd)
    scale = invstd * weight

    out = ((producer_f - centered * coeff) - mean_term) * scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


def _flat_view(t, total):
    if t.is_contiguous():
        return t.view(total)
    return t.as_strided((total,), (1,))


@oracle_impl(hardware="B200", point="ce2eb236")
def oracle_forward(inputs, **_kwargs):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
        _shape0, _shape1,
    ) = inputs
    device = arg0.device

    # ------------------------------------------------------------------
    # 1) view_1 = channel-shuffle(arg0, arg1) -> bf16[N, OUT_C, H, W] contig
    # ------------------------------------------------------------------
    view_1 = torch.empty((N, OUT_C, H, W), device=device, dtype=torch.bfloat16)
    view_1_flat = view_1.view(-1)

    # arg0 is contiguous NCHW.
    arg0_flat = arg0.contiguous().view(-1)
    # arg1 is channels-last NHWC memory. Flatten in memory order.
    arg1_flat = arg1.as_strided((N * C * HW,), (1,))

    total_out = N * OUT_C * HW
    max_block_shuffle = min(1024, _largest_pow2_divisor(total_out))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_out // max_block_shuffle, 1, 1),
        _shuffle_kernel,
        (arg0_flat, arg1_flat, view_1_flat, C, OUT_C, HW, max_block_shuffle),
    )

    # ------------------------------------------------------------------
    # 2) producer + partial per-channel reductions (per-tile, per-channel)
    # ------------------------------------------------------------------
    # slice_2 in NCHW-order contiguous [N, C, H, W]
    slice_2 = view_1[:, C:OUT_C, :, :].contiguous()  # NCHW contiguous
    slice_2_flat = slice_2.view(-1)  # N*C*HW

    fill_1d = arg7.reshape(1).contiguous()  # bf16 [1]
    mean_flat = arg3.reshape(C).contiguous()
    invstd_flat = arg4.reshape(C).contiguous()
    weight_flat = arg5.reshape(C).contiguous()
    bias_flat = arg6.reshape(C).contiguous()

    total_prod = N * C * HW
    NHW = N * HW
    # arg2 is channels-last NHWC memory.
    arg2_flat = arg2.as_strided((total_prod,), (1,))
    producer = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    producer_flat = _flat_view(producer, total_prod)

    # Partial-reduction grid parameters.
    BLOCK_R = 128  # NHW = 6272 is divisible by 128 (NHW // 128 = 49 tiles)
    BLOCK_C = 8    # C = 232 is divisible by 8 (232 // 8 = 29 col blocks)
    num_tiles = (NHW + BLOCK_R - 1) // BLOCK_R
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C

    partial_sum = torch.empty((num_tiles * C,), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles * C,), device=device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_tiles, num_c_blocks, 1),
        _producer_partial_kernel,
        (slice_2_flat, arg2_flat, fill_1d, mean_flat, invstd_flat, weight_flat,
         bias_flat, producer_flat, partial_sum, partial_dot,
         C, NHW, HW, BLOCK_R, BLOCK_C),
    )

    # ------------------------------------------------------------------
    # 3) finalize partials in-kernel with ct.sum (matches Triton fairness).
    # ------------------------------------------------------------------
    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((C,), device=device, dtype=torch.float32)
    mul_10 = torch.empty((C,), device=device, dtype=torch.float32)

    # BLOCK_TILES must be pow2 >= num_tiles.
    block_tiles = 1
    while block_tiles < num_tiles:
        block_tiles <<= 1
    FINAL_BLOCK_C = 8
    num_final_blocks = (C + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C

    ct.launch(
        stream,
        (num_final_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_flat,
         sum_1, sum_2, mul_10,
         C, num_tiles, block_tiles, FINAL_BLOCK_C),
    )

    # ------------------------------------------------------------------
    # 4) epilogue: BN-backward grad_input (channels-last)
    # ------------------------------------------------------------------
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    out_flat = _flat_view(out, total_prod)

    max_block = min(1024, _largest_pow2_divisor(total_prod))
    ct.launch(
        stream,
        (total_prod // max_block, 1, 1),
        _epilogue_nhwc_kernel,
        (producer_flat, arg2_flat, mean_flat, invstd_flat, weight_flat,
         sum_1, sum_2, out_flat, C, SCALE, max_block),
    )

    return view_1, sum_1, mul_10, out
