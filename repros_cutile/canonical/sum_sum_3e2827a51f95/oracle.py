"""cuTile port of sum_sum_3e2827a51f95: GhostNet BN backward tail (channels-last).

Mirrors Triton's 3-kernel structure with `ct.sum` inside the kernel:
  - `_partial_kernel`: (K_tiles, C_tiles) grid, BLOCK_K=1024, BLOCK_C=8;
    reduces N*H*W per channel to a (BLOCK_C,) partial via `ct.sum(..., axis=0)`.
  - `_finalize_kernel`: (C,) grid; sums 98 partial tiles per channel via `ct.sum`.
  - `_epilogue_kernel`: (num_elem_blocks,) grid; recomputes add, applies BN grad,
    stores channels-last bf16.

Channels=40, HW=196, N=512 -> K_TOTAL=N*HW=100352, TOTAL=4014080.
Everything is on the channels-last physical layout (NHWC contig views).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C0 = 80
C = 40
H = 14
W = 14
HW = H * W
K_TOTAL = N * HW      # 100352
TOTAL = N * C * HW    # 4014080
SCALE = 9.964923469387754e-06
N_TILES = 98          # K_TOTAL / 1024


def _ceil_pow2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _partial_kernel(
    slice_ptr,        # bf16 (K_TOTAL, C) — NHWC contig view of arg0[:, :40]
    arg1_ptr,         # bf16 (K_TOTAL, C) — NHWC contig view of arg1
    arg2_ptr,         # bf16 (K_TOTAL, C) — NHWC contig view of arg2
    mean_ptr,         # f32 (C,)
    partial_sum_ptr,  # f32 (N_TILES, C)
    partial_dot_ptr,  # f32 (N_TILES, C)
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)
    c_tile = ct.bid(1)
    x0 = ct.load(slice_ptr, index=(tile, c_tile), shape=(BLOCK_K, BLOCK_C))
    x1 = ct.load(arg1_ptr, index=(tile, c_tile), shape=(BLOCK_K, BLOCK_C))
    x0f = ct.astype(x0, ct.float32)
    x1f = ct.astype(x1, ct.float32)
    add_value = ct.astype(ct.astype(x0f + x1f, ct.bfloat16), ct.float32)

    x2 = ct.load(arg2_ptr, index=(tile, c_tile), shape=(BLOCK_K, BLOCK_C))
    x2f = ct.astype(x2, ct.float32)

    mean = ct.load(mean_ptr, index=(c_tile,), shape=(BLOCK_C,))  # (BLOCK_C,)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    centered = x2f - mean_2d
    prod = add_value * centered

    sum_part = ct.sum(add_value, axis=0)  # (BLOCK_C,)
    prod_part = ct.sum(prod, axis=0)

    ct.store(partial_sum_ptr, index=(tile, c_tile),
             tile=ct.reshape(sum_part, (1, BLOCK_C)))
    ct.store(partial_dot_ptr, index=(tile, c_tile),
             tile=ct.reshape(prod_part, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 (N_TILES, C)
    partial_dot_ptr,   # f32 (N_TILES, C)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)
    sum_out_ptr,       # f32 (C,)
    scale_grad_ptr,    # f32 (C,)
    mean_term_ptr,     # f32 (C,)
    prod_coeff_ptr,    # f32 (C,)
    output_scale_ptr,  # f32 (C,)
    BLOCK_TILES: ct.Constant[int],
):
    c = ct.bid(0)
    tile_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    active = tile_idx < N_TILES

    ps = ct.load(partial_sum_ptr, index=(0, c), shape=(BLOCK_TILES, 1),
                 padding_mode=ct.PaddingMode.ZERO)
    pd = ct.load(partial_dot_ptr, index=(0, c), shape=(BLOCK_TILES, 1),
                 padding_mode=ct.PaddingMode.ZERO)
    ps_1d = ct.reshape(ps, (BLOCK_TILES,))
    pd_1d = ct.reshape(pd, (BLOCK_TILES,))
    ps_active = ct.where(active, ps_1d, 0.0)
    pd_active = ct.where(active, pd_1d, 0.0)

    sum_value = ct.sum(ps_active)
    dot_value = ct.sum(pd_active)

    invstd_s = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_s = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_s, ())
    weight = ct.reshape(weight_s, ())

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(dot_value * invstd, (1,)))
    ct.store(mean_term_ptr, index=(c,), tile=ct.reshape(sum_value * SCALE, (1,)))
    ct.store(prod_coeff_ptr, index=(c,),
             tile=ct.reshape(dot_value * SCALE * invstd * invstd, (1,)))
    ct.store(output_scale_ptr, index=(c,),
             tile=ct.reshape(invstd * weight, (1,)))


@ct.kernel
def _epilogue_kernel(
    slice_ptr,         # bf16 (K_TOTAL, C)
    arg1_ptr,          # bf16 (K_TOTAL, C)
    arg2_ptr,          # bf16 (K_TOTAL, C)
    mean_ptr,          # f32 (C,)
    mean_term_ptr,     # f32 (C,)
    prod_coeff_ptr,    # f32 (C,)
    output_scale_ptr,  # f32 (C,)
    out_ptr,           # bf16 (K_TOTAL, C)
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)
    c_tile = ct.bid(1)
    x0 = ct.load(slice_ptr, index=(tile, c_tile), shape=(BLOCK_K, BLOCK_C))
    x1 = ct.load(arg1_ptr, index=(tile, c_tile), shape=(BLOCK_K, BLOCK_C))
    x0f = ct.astype(x0, ct.float32)
    x1f = ct.astype(x1, ct.float32)
    add_value = ct.astype(ct.astype(x0f + x1f, ct.bfloat16), ct.float32)

    x2 = ct.load(arg2_ptr, index=(tile, c_tile), shape=(BLOCK_K, BLOCK_C))
    x2f = ct.astype(x2, ct.float32)

    mean = ct.load(mean_ptr, index=(c_tile,), shape=(BLOCK_C,))
    mean_term = ct.load(mean_term_ptr, index=(c_tile,), shape=(BLOCK_C,))
    prod_coeff = ct.load(prod_coeff_ptr, index=(c_tile,), shape=(BLOCK_C,))
    output_scale = ct.load(output_scale_ptr, index=(c_tile,), shape=(BLOCK_C,))
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    prod_coeff_2d = ct.reshape(prod_coeff, (1, BLOCK_C))
    output_scale_2d = ct.reshape(output_scale, (1, BLOCK_C))

    centered = x2f - mean_2d
    out = (add_value - centered * prod_coeff_2d - mean_term_2d) * output_scale_2d
    out_bf = ct.astype(out, ct.bfloat16)
    ct.store(out_ptr, index=(tile, c_tile), tile=out_bf)


@oracle_impl(
    hardware="B200",
    point="453e4b44",
    BLOCK_K=1024,
    BLOCK_C=8,
)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    # Materialize contig NHWC views for cuTile kernels.
    # arg0_1[:, :40] is a channels-last strided view of arg0's 80-channel storage.
    # Permute(0,2,3,1).contiguous() gives (N, H, W, 40) contig — a copy.
    slice_nhwc = arg0_1[:, :C, :, :].permute(0, 2, 3, 1).contiguous()
    # arg1_1 is already channels-last (stride like (7840,1,560,40)); permute
    # gives contig NHWC without copy.
    arg1_nhwc = arg1_1.permute(0, 2, 3, 1).contiguous()
    arg2_nhwc = arg2_1.permute(0, 2, 3, 1).contiguous()

    slice_2d = slice_nhwc.view(K_TOTAL, C)
    arg1_2d = arg1_nhwc.view(K_TOTAL, C)
    arg2_2d = arg2_nhwc.view(K_TOTAL, C)
    mean_1d = arg3_1.view(C).contiguous()

    partial_sum = torch.empty((N_TILES, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((N_TILES, C), device=device, dtype=torch.float32)

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    num_c_tiles = C // BLOCK_C  # 5
    ct.launch(
        stream, (N_TILES, num_c_tiles, 1), _partial_kernel,
        (slice_2d, arg1_2d, arg2_2d, mean_1d,
         partial_sum, partial_dot,
         BLOCK_K, BLOCK_C),
    )
    ct.launch(
        stream, (C, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot,
         arg4_1.contiguous(), arg5_1.contiguous(),
         sum_out, scale_grad, mean_term, prod_coeff, output_scale,
         _ceil_pow2(N_TILES)),
    )

    # Output in channels-last physical order.
    out_nhwc = torch.empty((N, H, W, C), device=device, dtype=torch.bfloat16)
    out_2d = out_nhwc.view(K_TOTAL, C)
    ct.launch(
        stream, (N_TILES, num_c_tiles, 1), _epilogue_kernel,
        (slice_2d, arg1_2d, arg2_2d, mean_1d,
         mean_term, prod_coeff, output_scale,
         out_2d, BLOCK_K, BLOCK_C),
    )
    # Return as (N, C, H, W) with channels-last strides.
    out = out_nhwc.permute(0, 3, 1, 2)

    return sum_out, scale_grad, out
