"""cuTile port of sum_sum_ed2e3f6639e7: GhostNet BN backward.

Compute chain:
- affine = (arg1 - arg2) * arg3 * arg4 + arg5  (fp32, per-channel affine of arg1)
- relu on bf16(affine); le(relu <= 0) mask
- where mask: fill=arg6 else slice=arg0[:, 120:240]
- sum_1 = sum(where_f32) via in-kernel cuTile partial+finalize reduction
- sum_2 = sum(where_f32 * (arg1_f32 - arg2)) via same reduction pair
- BN backward: dense = ((where - centered * variance_term) - mean_term) * (arg3_scalar * arg4)
- Return (sum_1, mul_10=sum_2*arg3_scalar, dense_bf16)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C = 120
H = 28
W = 28
HW = H * W
NHW = N * HW  # 401408
SLICE_START = 120
SLICE_END = 240
SCALE = 2.4912308673469386e-06
PADDED = 1024

# Reduction tile shape (matches Triton reference _partial_reduce_kernel).
BLOCK_R = 512
BLOCK_C = 16
PADDED_C = ((C + BLOCK_C - 1) // BLOCK_C) * BLOCK_C  # 128
NUM_R_TILES = (NHW + BLOCK_R - 1) // BLOCK_R          # 784
NUM_C_TILES = PADDED_C // BLOCK_C                     # 8


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


BLOCK_TILES = _next_pow2(NUM_R_TILES)  # 1024


@ct.kernel
def _partial_reduce_kernel(
    where_ptr,        # fp32 (NHW, C) contiguous
    sub_ptr,          # fp32 (NHW, C) contiguous
    partial_sum_ptr,  # fp32 (NUM_R_TILES, PADDED_C) contiguous
    partial_dot_ptr,  # fp32 (NUM_R_TILES, PADDED_C) contiguous
    BLOCK_R_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    r_block = ct.bid(0)
    c_block = ct.bid(1)
    where_tile = ct.load(
        where_ptr,
        index=(r_block, c_block),
        shape=(BLOCK_R_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sub_tile = ct.load(
        sub_ptr,
        index=(r_block, c_block),
        shape=(BLOCK_R_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    partial_sum = ct.sum(where_tile, axis=0)              # (BLOCK_C_,)
    partial_dot = ct.sum(where_tile * sub_tile, axis=0)   # (BLOCK_C_,)
    ct.store(
        partial_sum_ptr,
        index=(r_block, c_block),
        tile=ct.reshape(partial_sum, (1, BLOCK_C_)),
    )
    ct.store(
        partial_dot_ptr,
        index=(r_block, c_block),
        tile=ct.reshape(partial_dot, (1, BLOCK_C_)),
    )


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # fp32 (NUM_R_TILES, PADDED_C)
    partial_dot_ptr,   # fp32 (NUM_R_TILES, PADDED_C)
    sum_out_ptr,       # fp32 (PADDED_C,)
    dot_out_ptr,       # fp32 (PADDED_C,)
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    ps = ct.load(
        partial_sum_ptr,
        index=(0, c_block),
        shape=(BLOCK_TILES_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pd = ct.load(
        partial_dot_ptr,
        index=(0, c_block),
        shape=(BLOCK_TILES_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_local = ct.arange(BLOCK_TILES_, dtype=ct.int32)
    tile_valid = tile_local < NUM_TILES
    mask = ct.reshape(tile_valid, (BLOCK_TILES_, 1))
    zero_2d = ct.zeros((BLOCK_TILES_, BLOCK_C_), dtype=ct.float32)
    ps_masked = ct.where(mask, ps, zero_2d)
    pd_masked = ct.where(mask, pd, zero_2d)
    sum_value = ct.sum(ps_masked, axis=0)   # (BLOCK_C_,)
    dot_value = ct.sum(pd_masked, axis=0)   # (BLOCK_C_,)
    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(dot_out_ptr, index=(c_block,), tile=dot_value)


@ct.kernel
def _bn_backward_dense_kernel(
    where_ptr, centered_ptr, variance_ptr, mean_term_ptr, output_scale_ptr,
    dense_ptr,
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    where_f = ct.load(where_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW))
    centered = ct.load(centered_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW))
    variance_term = ct.load(variance_ptr, index=(c,), shape=(1,))
    mean_term = ct.load(mean_term_ptr, index=(c,), shape=(1,))
    output_scale = ct.load(output_scale_ptr, index=(c,), shape=(1,))
    var_1 = ct.reshape(variance_term, (1, 1, 1))
    mean_1 = ct.reshape(mean_term, (1, 1, 1))
    scale_1 = ct.reshape(output_scale, (1, 1, 1))
    corrected = where_f - centered * var_1
    centered_grad = corrected - mean_1
    dense = centered_grad * scale_1
    dense_bf = ct.astype(dense, ct.bfloat16)
    ct.store(dense_ptr, index=(n, c, 0), tile=dense_bf)


@oracle_impl(hardware="B200", point="e260e21e", BLOCK_HW=PADDED)
def oracle_forward(inputs, *, BLOCK_HW: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    device = arg1_1.device

    slice_1 = arg0_1[:, SLICE_START:SLICE_END]  # bf16 slice of arg0

    # Compute affine + relu -> le mask
    sub_affine = arg1_1.to(torch.float32) - arg2_1
    mul_affine = sub_affine * arg3_1 * arg4_1.view(1, C, 1, 1) + arg5_1.view(1, C, 1, 1)
    affine_bf16 = mul_affine.to(torch.bfloat16)
    relu_bf16 = torch.relu(affine_bf16)
    le = relu_bf16 <= 0.0

    where_bf = torch.where(le, arg6_1.to(torch.bfloat16), slice_1)
    where_f = where_bf.to(torch.float32)

    arg3_flat = arg3_1.view(C)

    sub_1 = arg1_1.to(torch.float32) - arg2_1  # [N, C, H, W]

    # === In-kernel cuTile reductions: sum_1, sum_2 over (N, H, W) per channel. ===
    # Reshape to (NHW, C) contiguous so partial-reduce tiles are dense loads.
    where_hwc = where_f.permute(0, 2, 3, 1).contiguous().view(NHW, C)
    sub_hwc = sub_1.permute(0, 2, 3, 1).contiguous().view(NHW, C)

    partial_sum = torch.empty(
        (NUM_R_TILES, PADDED_C), device=device, dtype=torch.float32
    )
    partial_dot = torch.empty(
        (NUM_R_TILES, PADDED_C), device=device, dtype=torch.float32
    )
    sum_out_padded = torch.empty((PADDED_C,), device=device, dtype=torch.float32)
    dot_out_padded = torch.empty((PADDED_C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUM_R_TILES, NUM_C_TILES, 1),
        _partial_reduce_kernel,
        (where_hwc, sub_hwc, partial_sum, partial_dot, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (NUM_C_TILES, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, sum_out_padded, dot_out_padded,
         NUM_R_TILES, BLOCK_TILES, BLOCK_C),
    )
    sum_1 = sum_out_padded[:C]
    sum_2 = dot_out_padded[:C]

    mul_3 = sum_1 * SCALE
    mul_4 = sum_2 * SCALE
    variance_term = mul_4 * (arg3_flat * arg3_flat)
    output_scale = arg3_flat * arg4_1

    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    where_pad = torch.zeros((N, C, PADDED), device=device, dtype=torch.float32)
    where_pad[..., :HW].copy_(where_f.view(N, C, HW))
    sub_pad = torch.zeros((N, C, PADDED), device=device, dtype=torch.float32)
    sub_pad[..., :HW].copy_(sub_1.view(N, C, HW))
    dense_pad = torch.empty((N, C, PADDED), device=device, dtype=torch.bfloat16)

    ct.launch(
        stream, (N, C, 1), _bn_backward_dense_kernel,
        (where_pad, sub_pad, variance_term, mul_3, output_scale, dense_pad, BLOCK_HW),
    )
    dense_out.view(N, C, HW).copy_(dense_pad[..., :HW])

    mul_10 = sum_2 * arg3_flat

    return sum_1, mul_10, dense_out
