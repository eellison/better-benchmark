"""cuTile port of sum_sum_2b33da3550b0: DenseNet BN-backward + pool-backward.

Multi-kernel strategy:
  1. Per-channel reduction kernel: for each channel c, compute
     sum_over_batch_hw(selected) and sum_over_batch_hw(selected * centered)
     where selected = where(mask <= 0, fill, source), centered = activation-mean.
  2. Finalize per-channel scalars in torch.
  3. Pool-backward kernel: for each (n, c) pair, compute per-spatial grad, add
     residual slice, /4, then store 4-way expanded pool output (2x2 duplication).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
INPUT_C = 80
C = 64
H = 16
W = 16
OUT_H = 32
OUT_W = 32
HW = H * W
OUT_HW = OUT_H * OUT_W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
SLICE_START = 16
SCALE = 3.0517578125e-05


@ct.kernel
def _dual_reduce_kernel(
    mask_input_ptr,      # bf16 [N, C, H, W] compact
    fill_ptr,            # bf16 []
    source_ptr,          # bf16 [N, C, H, W] compact
    centered_source_ptr, # bf16 [N, C, H, W] compact
    mean_ptr,            # f32  [1, C, 1, 1]  -> [C]
    partial_sum_ptr,     # f32  [C]
    partial_dot_ptr,     # f32  [C]
    N_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    # Process a tile of (batch, spatial) at a time; we iterate the whole thing
    # in the kernel. Here we run a single tile per (c, tile) and accumulate
    # partial sums via atomic_add. Actually simpler: run one tile per (c, tile)
    # and store to (c, tile) partial slot.
    #
    # But cleanest: launch grid = (C, num_tiles_hwn), each writes to
    # partial arrays indexed via atomic_add.

    hw_offsets = ct.arange(BLOCK_HW, dtype=ct.int32) + tile * BLOCK_HW
    active = hw_offsets < (N_ * HW_)

    n = hw_offsets // HW_
    hw = hw_offsets - n * HW_

    # Load fill and mean scalars
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))

    # Build 1D flat index: n*C*HW + c*HW + hw. Use gather via ct.gather? Simpler:
    # cuTile supports multi-dim indexing on 4D tensors. Use gather.
    n_tile = n
    hw_tile = hw
    c_tile = ct.full((BLOCK_HW,), c, dtype=ct.int32)

    # Treat everything as 4D [N, C, HW, 1] with hw flattened, but the original
    # is [N, C, H, W]. Use 3D view: reshape mask_input to [N, C, HW] then gather.
    # Actually since 4D gather works, use it: index tuple (n, c, h, w) where h=hw//W, w=hw%W.
    h_tile = hw // W
    w_tile = hw - h_tile * W

    mask_v = ct.gather(mask_input_ptr, (n_tile, c_tile, h_tile, w_tile), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n_tile, c_tile, h_tile, w_tile), mask=active,
                         padding_value=ct.bfloat16(0.0))
    cen_src_v = ct.gather(centered_source_ptr, (n_tile, c_tile, h_tile, w_tile), mask=active,
                          padding_value=ct.bfloat16(0.0))

    fill_v = ct.full((BLOCK_HW,), 0.0, dtype=ct.bfloat16) + ct.reshape(fill, (1,))
    selected_bf = ct.where(mask_v <= ct.astype(0.0, ct.bfloat16), fill_v, source_v)
    selected = ct.astype(selected_bf, ct.float32)
    cen_src = ct.astype(cen_src_v, ct.float32)
    centered = cen_src - ct.astype(mean, ct.float32)
    product = selected * centered

    sel_masked = ct.where(active, selected, 0.0)
    prod_masked = ct.where(active, product, 0.0)
    sum_val = ct.sum(sel_masked)
    dot_val = ct.sum(prod_masked)
    ct.atomic_add(partial_sum_ptr, c, sum_val)
    ct.atomic_add(partial_dot_ptr, c, dot_val)


@ct.kernel
def _pool_epilogue_kernel(
    residual_ptr,        # bf16 [N, INPUT_C, H, W]
    mask_input_ptr,      # bf16 [N, C, H, W]
    fill_ptr,            # bf16 []
    source_ptr,          # bf16 [N, C, H, W]
    centered_source_ptr, # bf16 [N, C, H, W]
    mean_ptr,            # f32  [C]
    mean_term_ptr,       # f32  [C]
    variance_scale_ptr,  # f32  [C]
    output_scale_ptr,    # f32  [C]
    pool_out_ptr,        # bf16 [N, C, OUT_H, OUT_W]
    HW_: ct.Constant[int],
    W_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    OUT_W_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    # Grid: (N, C, HW/BLOCK) covers every (n,c) then walks the HW dimension in BLOCK tiles.
    n = ct.bid(0)
    c = ct.bid(1)
    tile = ct.bid(2)
    hw_off = ct.arange(BLOCK, dtype=ct.int32) + tile * BLOCK
    active = hw_off < HW_
    h = hw_off // W_
    w = hw_off - h * W_

    n_t = ct.full((BLOCK,), n, dtype=ct.int32)
    c_t = ct.full((BLOCK,), c, dtype=ct.int32)
    c_res_t = ct.full((BLOCK,), c + SLICE_START_, dtype=ct.int32)

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_term = ct.load(mean_term_ptr, index=(c,), shape=(1,))
    var_scale = ct.load(variance_scale_ptr, index=(c,), shape=(1,))
    out_scale = ct.load(output_scale_ptr, index=(c,), shape=(1,))

    mask_v = ct.gather(mask_input_ptr, (n_t, c_t, h, w), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n_t, c_t, h, w), mask=active,
                         padding_value=ct.bfloat16(0.0))
    cen_src_v = ct.gather(centered_source_ptr, (n_t, c_t, h, w), mask=active,
                          padding_value=ct.bfloat16(0.0))
    residual = ct.gather(residual_ptr, (n_t, c_res_t, h, w), mask=active,
                         padding_value=ct.bfloat16(0.0))

    fill_v = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16) + ct.reshape(fill, (1,))
    selected_bf = ct.where(mask_v <= ct.astype(0.0, ct.bfloat16), fill_v, source_v)
    selected = ct.astype(selected_bf, ct.float32)
    cen_src = ct.astype(cen_src_v, ct.float32)

    mean_bcast = ct.broadcast_to(ct.astype(mean, ct.float32), (BLOCK,))
    mean_term_bcast = ct.broadcast_to(ct.astype(mean_term, ct.float32), (BLOCK,))
    var_scale_bcast = ct.broadcast_to(ct.astype(var_scale, ct.float32), (BLOCK,))
    out_scale_bcast = ct.broadcast_to(ct.astype(out_scale, ct.float32), (BLOCK,))

    centered = cen_src - mean_bcast
    after_variance = selected - centered * var_scale_bcast
    after_mean = after_variance - mean_term_bcast
    grad_bf = ct.astype(after_mean * out_scale_bcast, ct.bfloat16)

    add_bf = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(grad_bf, ct.float32),
        ct.bfloat16,
    )
    pool_val = ct.astype(ct.astype(add_bf, ct.float32) * 0.25, ct.bfloat16)

    # Store pool_val to (n, c, 2h, 2w), (n, c, 2h, 2w+1), (n, c, 2h+1, 2w), (n, c, 2h+1, 2w+1)
    oh = h * 2
    ow = w * 2
    ct.scatter(pool_out_ptr, (n_t, c_t, oh, ow), pool_val, mask=active)
    ct.scatter(pool_out_ptr, (n_t, c_t, oh, ow + 1), pool_val, mask=active)
    ct.scatter(pool_out_ptr, (n_t, c_t, oh + 1, ow), pool_val, mask=active)
    ct.scatter(pool_out_ptr, (n_t, c_t, oh + 1, ow + 1), pool_val, mask=active)


@oracle_impl(hardware="B200", point="cc6a993c", BLOCK_K=1024, BLOCK_ELEMS=256)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_ELEMS: int):
    (
        arg0_1,   # bf16 [128, 80, 16, 16] — residual
        arg1_1,   # bf16 [128, 64, 16, 16] — mask
        arg2_1,   # bf16 []                 — fill
        arg3_1,   # bf16 [128, 64, 16, 16] — source (rhs)
        arg4_1,   # bf16 [128, 64, 16, 16] — centered_source (activation)
        arg5_1,   # f32  [1, 64, 1, 1]    — mean
        arg6_1,   # f32  [64]              — invstd
        arg7_1,   # f32  [64]              — weight
        _arg8_1,
    ) = inputs
    device = arg0_1.device

    # Partial (per-channel) sums via atomic_add
    partial_sum = torch.zeros((C,), device=device, dtype=torch.float32)
    partial_dot = torch.zeros((C,), device=device, dtype=torch.float32)

    mean_1d = arg5_1.view(C)
    fill_1d = arg2_1.view(1)

    stream = torch.cuda.current_stream()
    num_tiles_hwn = (TOTAL_SPATIAL + BLOCK_K - 1) // BLOCK_K
    ct.launch(
        stream,
        (C, num_tiles_hwn, 1),
        _dual_reduce_kernel,
        (arg1_1, fill_1d, arg3_1, arg4_1, mean_1d,
         partial_sum, partial_dot, N, HW, BLOCK_K),
    )

    # Finalize per-channel scalars via torch (fine — captureable).
    invstd = arg6_1
    weight = arg7_1
    scale_grad = partial_dot * invstd
    mean_term = partial_sum * SCALE
    dot_mean = partial_dot * SCALE
    variance_scale = dot_mean * invstd * invstd
    output_scale = invstd * weight

    pool_out = torch.empty_strided(
        (N, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # Choose per-(n,c) HW tile block
    hw_block = HW  # 256 fits in one tile
    ct.launch(
        stream,
        (N, C, (HW + hw_block - 1) // hw_block),
        _pool_epilogue_kernel,
        (arg0_1, arg1_1, fill_1d, arg3_1, arg4_1,
         mean_1d, mean_term, variance_scale, output_scale,
         pool_out, HW, W, SLICE_START, OUT_W, hw_block),
    )

    return partial_sum, scale_grad, pool_out
