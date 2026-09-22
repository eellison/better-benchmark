"""cuTile port of sum_sum_ed4730568670: ShuffleNet BN backward with channel shuffle.

Channel shuffle + BN affine + relu + where are structural / elementwise
non-reduction ops and stay in torch. The two channel `sum(...)` reductions
that Triton fuses (`sum_1 = sum(conv_1)` and `sum_2 = sum(conv_1 * sub_1)`
over dims (0, 2, 3)) move into cuTile as a split-K partial reduce +
per-channel finalize pair. The final elementwise BN backward remains in a
cuTile epilogue kernel.

This mirrors Triton's 3-kernel structure: partial_reduce (per c, per k-tile)
-> finalize (per c) -> bn-backward epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
C_HALF = 58
C_TOTAL = 116
H = 28
W = 28
INV_N = 9.964923469387754e-06
PIXELS = BATCH * C_HALF * H * W
NHW = BATCH * H * W  # 100352

BLOCK = 1024
BLOCK_K = 1024
BLOCK_C = 1


@ct.kernel
def _partial_reduce_kernel(
    conv_1_ptr,       # f32 [N, C, HW]
    sub_1_ptr,        # f32 [N, C, HW]
    partial_sum_ptr,  # f32 [num_tiles, C]
    partial_dot_ptr,  # f32 [num_tiles, C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    NHW_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
):
    """Grid: (C, num_tiles, 1). One (c, k-tile) partial per program.

    Computes sum(conv_1) and sum(conv_1 * sub_1) over the BLOCK_K_ slice
    of N*HW for the fixed channel c.
    """
    c = ct.bid(0)
    tile = ct.bid(1)
    k_idx = ct.arange(BLOCK_K_, dtype=ct.int32)
    k_global = tile * BLOCK_K_ + k_idx
    active = k_global < NHW_
    n = k_global // HW_
    hw = k_global - n * HW_
    c_bc = ct.full((BLOCK_K_,), c, dtype=ct.int32)

    conv_1 = ct.gather(conv_1_ptr, (n, c_bc, hw), mask=active,
                       padding_value=0.0)
    sub_1 = ct.gather(sub_1_ptr, (n, c_bc, hw), mask=active,
                      padding_value=0.0)
    conv_1 = ct.where(active, conv_1, 0.0)
    sub_1 = ct.where(active, sub_1, 0.0)

    partial_sum = ct.sum(conv_1)
    partial_dot = ct.sum(conv_1 * sub_1)
    ct.store(partial_sum_ptr, index=(tile, c),
             tile=ct.reshape(partial_sum, (1, 1)))
    ct.store(partial_dot_ptr, index=(tile, c),
             tile=ct.reshape(partial_dot, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,  # f32 [num_tiles, C]
    partial_dot_ptr,  # f32 [num_tiles, C]
    sum_out_ptr,      # f32 [C]  == sum_1
    dot_out_ptr,      # f32 [C]  == sum_2
    C_: ct.Constant[int],
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    """Grid: (num_c_tiles, 1, 1). Reduces partial rows along tile axis for
    BLOCK_C_ channels per program.
    """
    c_block = ct.bid(0)
    c_local = ct.arange(BLOCK_C_, dtype=ct.int32)
    c_idx = c_block * BLOCK_C_ + c_local
    c_valid = c_idx < C_

    tile_local = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tile_valid = tile_local < NUM_TILES
    mask = ct.reshape(tile_valid, (BLOCK_TILES, 1)) & ct.reshape(
        c_valid, (1, BLOCK_C_))
    zero_2d = ct.zeros((BLOCK_TILES, BLOCK_C_), dtype=ct.float32)

    ps = ct.load(
        partial_sum_ptr, index=(0, c_block),
        shape=(BLOCK_TILES, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pd = ct.load(
        partial_dot_ptr, index=(0, c_block),
        shape=(BLOCK_TILES, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ps_masked = ct.where(mask, ps, zero_2d)
    pd_masked = ct.where(mask, pd, zero_2d)
    sum_value = ct.sum(ps_masked, axis=0)  # (BLOCK_C_,)
    dot_value = ct.sum(pd_masked, axis=0)

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(dot_out_ptr, index=(c_block,), tile=dot_value)


@ct.kernel
def _bn_backward_elem_kernel(
    conv_1_ptr,      # f32 [PIXELS]
    sub_1_ptr,       # f32 [PIXELS]
    scale_ptr,       # f32 [C]  (mul_7 = squeeze * arg5)
    factor_ptr,      # f32 [C]  (mul_3 = sum_1 * INV_N)
    var_scale_ptr,   # f32 [C]  (mul_6 = sum_2*INV_N*squeeze^2)
    out_ptr,         # bf16 [PIXELS]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = ct.arange(BLOCK, dtype=ct.int64) + pid * BLOCK
    conv_1 = ct.load(conv_1_ptr, index=(pid,), shape=(BLOCK,))
    sub_1 = ct.load(sub_1_ptr, index=(pid,), shape=(BLOCK,))
    c = (idx // HW) % C
    scale = ct.gather(scale_ptr, c)
    factor = ct.gather(factor_ptr, c)
    var_scale = ct.gather(var_scale_ptr, c)
    mul_8 = sub_1 * var_scale
    sub_2 = conv_1 - mul_8
    sub_3 = sub_2 - factor
    mul_9 = sub_3 * scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(mul_9, ct.bfloat16))


@oracle_impl(hardware="B200", point="94013926")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     shape0, shape1) = inputs
    device = arg0_1.device

    # Channel shuffle (structural — no math).
    slice_1 = arg0_1[:, :C_HALF, :, :]
    cat = torch.cat([slice_1, arg1_1], dim=1)  # bf16 [128, 116, 28, 28]
    shape0_t = tuple(int(d) for d in shape0)
    shape1_t = tuple(int(d) for d in shape1)
    view = cat.view(shape0_t)
    permute = view.permute(0, 2, 1, 3, 4)
    clone = permute.contiguous()
    view_1 = clone.view(shape1_t)
    slice_2 = view_1[:, C_HALF:C_TOTAL, :, :]

    # BN affine forward + relu + where (elementwise, keep in torch).
    sub = arg2_1 - arg3_1
    mul = sub * arg4_1
    mul_1 = mul * arg5_1.view(1, C_HALF, 1, 1)
    add = mul_1 + arg6_1.view(1, C_HALF, 1, 1)
    conv = add.to(torch.bfloat16)
    relu = torch.relu(conv)
    le = relu <= 0
    where = torch.where(le, arg7_1, slice_2)
    conv_1 = where.to(torch.float32)

    squeeze = arg3_1.squeeze(0).squeeze(-1).squeeze(-1)  # [C]
    conv_2 = arg2_1.to(torch.float32)
    sub_1 = conv_2 - squeeze.view(1, C_HALF, 1, 1)

    # --- cuTile split-K partial reduce + finalize for sum_1 and sum_2. ---
    conv_1_c = conv_1.contiguous()
    sub_1_c = sub_1.contiguous()
    conv_1_3d = conv_1_c.view(BATCH, C_HALF, H * W)
    sub_1_3d = sub_1_c.view(BATCH, C_HALF, H * W)

    num_tiles = (NHW + BLOCK_K - 1) // BLOCK_K
    partial_sum = torch.empty((num_tiles, C_HALF),
                              device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, C_HALF),
                              device=device, dtype=torch.float32)
    sum_1 = torch.empty((C_HALF,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((C_HALF,), device=device, dtype=torch.float32)

    block_tiles = 1
    while block_tiles < num_tiles:
        block_tiles *= 2
    num_c_tiles = (C_HALF + BLOCK_C - 1) // BLOCK_C

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C_HALF, num_tiles, 1), _partial_reduce_kernel,
        (conv_1_3d, sub_1_3d, partial_sum, partial_dot,
         C_HALF, H * W, NHW, BLOCK_K),
    )
    ct.launch(
        stream, (num_c_tiles, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, sum_1, sum_2,
         C_HALF, num_tiles, block_tiles, BLOCK_C),
    )

    # Derived per-channel quantities feeding the BN-backward epilogue.
    mul_3 = sum_1 * INV_N
    squeeze_1 = arg4_1.squeeze(0).squeeze(-1).squeeze(-1)
    mul_5 = squeeze_1 * squeeze_1
    mul_4 = sum_2 * INV_N
    mul_6 = mul_4 * mul_5
    mul_7 = squeeze_1 * arg5_1
    mul_10 = sum_2 * squeeze_1  # returned

    # Elementwise BN backward in cuTile.
    conv_1_flat = conv_1_c.view(-1)
    sub_1_flat = sub_1_c.view(-1)
    out = torch.empty_strided(
        (BATCH, C_HALF, H, W), (C_HALF * H * W, H * W, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_flat = out.view(-1)

    ct.launch(
        stream, (ct.cdiv(PIXELS, BLOCK), 1, 1), _bn_backward_elem_kernel,
        (conv_1_flat, sub_1_flat, mul_7.contiguous(),
         mul_3.contiguous(), mul_6.contiguous(), out_flat,
         C_HALF, H * W, BLOCK),
    )
    return view_1, sum_1, mul_10, out
