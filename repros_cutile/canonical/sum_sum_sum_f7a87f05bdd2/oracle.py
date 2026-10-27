"""cuTile port of sum_sum_sum_f7a87f05bdd2: GhostNet dual BN-backward with
channels-last slice reductions.

Structure mirrors Triton's 4-kernel plan; channel reductions (sum_1..sum_4)
are computed IN-kernel via cuTile primitives (ct.sum) via a partial+finalize
split matching Triton's `_partial_reductions_kernel` + `_finalize_reductions_kernel`:

  1) _add_bf16_kernel: sum arg0+arg1 -> contiguous bf16 buffer.
  2) _partial_channel_reduce_kernel: per (n, hw_tile) tile, sum over rows to
     produce partials for sum_1/sum_2 (C40) and sum_3/sum_4 (C20 slice).
  3) _finalize_reduce_kernel: reduce partials across tiles -> C40/C20 finals.
  4) _grad_row_kernel: bf16 gradient rows via BN-backward math.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C40 = 40
C20 = 20
H = 28
W = 28
HW = H * W  # 784
HW_PAD = 1024
INV_ROWS = 2.4912308673469386e-06  # 1 / (N*H*W)

# Partial reduction plan: N*HW = 512 * 784 = 401408 rows.
# Split into partial tiles of ROWS_PER_TILE.
ROWS_PER_TILE = 1024
NUM_TILES = (N * HW + ROWS_PER_TILE - 1) // ROWS_PER_TILE  # = 392


@ct.kernel
def _add_bf16_kernel(
    a_ptr,       # bf16 [TOTAL]
    b_ptr,       # bf16 [TOTAL]
    out_ptr,     # bf16 [TOTAL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    a = ct.load(a_ptr, index=(pid,), shape=(BLOCK,))
    b = ct.load(b_ptr, index=(pid,), shape=(BLOCK,))
    add = ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=add)


@ct.kernel
def _partial_reductions_kernel(
    added_ptr,       # bf16 [N*HW, C40]  (channels-last view)
    arg2_ptr,        # bf16 [N*HW, C40]
    arg3_ptr,        # f32  [C40]
    slice_ptr,       # bf16 [N*HW, C20]  = added[:, 20:40] contiguous
    arg6_ptr,        # bf16 [N*HW, C20]
    arg7_ptr,        # f32  [C20]
    partial_x40_ptr,     # f32 [NUM_TILES, C40]
    partial_xrhs40_ptr,  # f32 [NUM_TILES, C40]
    partial_x20_ptr,     # f32 [NUM_TILES, C20]
    partial_xrhs20_ptr,  # f32 [NUM_TILES, C20]
    ROWS_TOTAL: ct.Constant[int],
    ROWS_PER_TILE_: ct.Constant[int],
    C40_PAD: ct.Constant[int],
    C20_PAD: ct.Constant[int],
    C40_: ct.Constant[int],
    C20_: ct.Constant[int],
):
    tile = ct.bid(0)

    # Load a [ROWS_PER_TILE, C40_PAD] tile of the CL producer.
    x40 = ct.load(added_ptr, index=(tile, 0), shape=(ROWS_PER_TILE_, C40_PAD),
                  padding_mode=ct.PaddingMode.ZERO)
    x40_f = ct.astype(x40, ct.float32)
    centered_arg = ct.load(arg2_ptr, index=(tile, 0), shape=(ROWS_PER_TILE_, C40_PAD),
                           padding_mode=ct.PaddingMode.ZERO)
    centered_f = ct.astype(centered_arg, ct.float32)
    arg3 = ct.astype(ct.load(arg3_ptr, index=(0,), shape=(C40_PAD,),
                             padding_mode=ct.PaddingMode.ZERO), ct.float32)
    arg3_2d = ct.reshape(arg3, (1, C40_PAD))
    centered40 = centered_f - arg3_2d

    # Row mask (rows < ROWS_TOTAL) — needed if the tile spans past-end rows.
    row_idx = tile * ROWS_PER_TILE_ + ct.arange(ROWS_PER_TILE_, dtype=ct.int32)
    row_valid = ct.reshape(row_idx < ROWS_TOTAL, (ROWS_PER_TILE_, 1))
    # C40 col mask.
    c40_idx = ct.arange(C40_PAD, dtype=ct.int32)
    c40_valid = ct.reshape(c40_idx < C40_, (1, C40_PAD))
    valid40 = row_valid & c40_valid
    zeros40 = ct.full((ROWS_PER_TILE_, C40_PAD), 0.0, dtype=ct.float32)
    x40_m = ct.where(valid40, x40_f, zeros40)

    sum_x40 = ct.sum(x40_m, axis=0)  # (C40_PAD,)
    sum_xrhs40 = ct.sum(x40_m * centered40, axis=0)
    ct.store(partial_x40_ptr, index=(tile, 0),
             tile=ct.reshape(sum_x40, (1, C40_PAD)))
    ct.store(partial_xrhs40_ptr, index=(tile, 0),
             tile=ct.reshape(sum_xrhs40, (1, C40_PAD)))

    # C20 slice: pre-materialized contiguous [N*HW, C20]
    x20 = ct.load(slice_ptr, index=(tile, 0), shape=(ROWS_PER_TILE_, C20_PAD),
                  padding_mode=ct.PaddingMode.ZERO)
    x20_f = ct.astype(x20, ct.float32)
    arg6 = ct.load(arg6_ptr, index=(tile, 0), shape=(ROWS_PER_TILE_, C20_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    arg6_f = ct.astype(arg6, ct.float32)
    arg7 = ct.astype(ct.load(arg7_ptr, index=(0,), shape=(C20_PAD,),
                             padding_mode=ct.PaddingMode.ZERO), ct.float32)
    arg7_2d = ct.reshape(arg7, (1, C20_PAD))
    centered20 = arg6_f - arg7_2d

    c20_idx = ct.arange(C20_PAD, dtype=ct.int32)
    c20_valid = ct.reshape(c20_idx < C20_, (1, C20_PAD))
    valid20 = row_valid & c20_valid
    zeros20 = ct.full((ROWS_PER_TILE_, C20_PAD), 0.0, dtype=ct.float32)
    x20_m = ct.where(valid20, x20_f, zeros20)

    sum_x20 = ct.sum(x20_m, axis=0)
    sum_xrhs20 = ct.sum(x20_m * centered20, axis=0)
    ct.store(partial_x20_ptr, index=(tile, 0),
             tile=ct.reshape(sum_x20, (1, C20_PAD)))
    ct.store(partial_xrhs20_ptr, index=(tile, 0),
             tile=ct.reshape(sum_xrhs20, (1, C20_PAD)))


@ct.kernel
def _finalize_reduce_kernel(
    partial_x40_ptr,     # f32 [NUM_TILES, C40]
    partial_xrhs40_ptr,  # f32 [NUM_TILES, C40]
    partial_x20_ptr,     # f32 [NUM_TILES, C20]
    partial_xrhs20_ptr,  # f32 [NUM_TILES, C20]
    arg4_ptr,            # f32 [C40]
    arg8_ptr,            # f32 [C20]
    sum1_ptr,            # f32 [C40]
    mul8_ptr,            # f32 [C40]
    sum3_ptr,            # f32 [C20]
    mul17_ptr,           # f32 [C20]
    NUM_TILES_: ct.Constant[int],
    NUM_TILES_PAD: ct.Constant[int],
    C40_PAD: ct.Constant[int],
    C20_PAD: ct.Constant[int],
    C40_: ct.Constant[int],
    C20_: ct.Constant[int],
):
    tile_idx = ct.arange(NUM_TILES_PAD, dtype=ct.int32)
    tile_valid = ct.reshape(tile_idx < NUM_TILES_, (NUM_TILES_PAD, 1))

    c40_idx = ct.arange(C40_PAD, dtype=ct.int32)
    c40_valid = ct.reshape(c40_idx < C40_, (1, C40_PAD))
    valid40 = tile_valid & c40_valid
    zeros40 = ct.full((NUM_TILES_PAD, C40_PAD), 0.0, dtype=ct.float32)

    p40 = ct.load(partial_x40_ptr, index=(0, 0),
                  shape=(NUM_TILES_PAD, C40_PAD),
                  padding_mode=ct.PaddingMode.ZERO)
    pr40 = ct.load(partial_xrhs40_ptr, index=(0, 0),
                   shape=(NUM_TILES_PAD, C40_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    p40_m = ct.where(valid40, p40, zeros40)
    pr40_m = ct.where(valid40, pr40, zeros40)
    sum1 = ct.sum(p40_m, axis=0)
    sum2 = ct.sum(pr40_m, axis=0)
    arg4 = ct.load(arg4_ptr, index=(0,), shape=(C40_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    mul8 = sum2 * arg4
    ct.store(sum1_ptr, index=(0,), tile=sum1)
    ct.store(mul8_ptr, index=(0,), tile=mul8)

    c20_idx = ct.arange(C20_PAD, dtype=ct.int32)
    c20_valid = ct.reshape(c20_idx < C20_, (1, C20_PAD))
    valid20 = tile_valid & c20_valid
    zeros20 = ct.full((NUM_TILES_PAD, C20_PAD), 0.0, dtype=ct.float32)

    p20 = ct.load(partial_x20_ptr, index=(0, 0),
                  shape=(NUM_TILES_PAD, C20_PAD),
                  padding_mode=ct.PaddingMode.ZERO)
    pr20 = ct.load(partial_xrhs20_ptr, index=(0, 0),
                   shape=(NUM_TILES_PAD, C20_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    p20_m = ct.where(valid20, p20, zeros20)
    pr20_m = ct.where(valid20, pr20, zeros20)
    sum3 = ct.sum(p20_m, axis=0)
    sum4 = ct.sum(pr20_m, axis=0)
    arg8 = ct.load(arg8_ptr, index=(0,), shape=(C20_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    mul17 = sum4 * arg8
    ct.store(sum3_ptr, index=(0,), tile=sum3)
    ct.store(mul17_ptr, index=(0,), tile=mul17)


@ct.kernel
def _grad_row_kernel(
    copy1_ptr,      # bf16 [N*C, HW] contiguous
    arg_ptr,        # bf16 [N*C, HW] contiguous (arg2 or arg6 flat)
    arg_center_ptr, # f32 [C]  (arg3 or arg7 flat)
    coef_a_ptr,     # f32 [C]
    coef_b_ptr,     # f32 [C]
    coef_c_ptr,     # f32 [C]
    out_grad_ptr,   # bf16 output — indexed by (row, col) with mask
    HW_: ct.Constant[int],
    HW_PAD_: ct.Constant[int],
    C_: ct.Constant[int],
):
    nc = ct.bid(0)  # flat n*C + c
    # Compute c = nc % C using arithmetic.
    n = nc // C_
    c = nc - n * C_

    cols = ct.arange(HW_PAD_, dtype=ct.int32)
    valid = cols < HW_

    x = ct.load(
        copy1_ptr, index=(nc, 0), shape=(1, HW_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_center = ct.load(
        arg_ptr, index=(nc, 0), shape=(1, HW_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Load per-channel scalars
    arg_c_1 = ct.load(arg_center_ptr, index=(c,), shape=(1,))
    coef_a_1 = ct.load(coef_a_ptr, index=(c,), shape=(1,))
    coef_b_1 = ct.load(coef_b_ptr, index=(c,), shape=(1,))
    coef_c_1 = ct.load(coef_c_ptr, index=(c,), shape=(1,))

    x_f = ct.astype(x, ct.float32)
    x_center_f = ct.astype(x_center, ct.float32)
    arg_c_2d = ct.reshape(arg_c_1, (1, 1))
    coef_a_2d = ct.reshape(coef_a_1, (1, 1))
    coef_b_2d = ct.reshape(coef_b_1, (1, 1))
    coef_c_2d = ct.reshape(coef_c_1, (1, 1))

    centered = x_center_f - arg_c_2d
    inner = x_f - centered * coef_a_2d - coef_b_2d
    out = inner * coef_c_2d
    out_bf = ct.astype(out, ct.bfloat16)

    # Scatter with mask into out_grad_ptr shape [N*C, HW].
    row_bcast = ct.full((1, HW_PAD_), nc, dtype=ct.int32)
    valid_2d = ct.reshape(valid, (1, HW_PAD_))
    cols_2d = ct.reshape(cols, (1, HW_PAD_))
    ct.scatter(out_grad_ptr, (row_bcast, cols_2d), out_bf, mask=valid_2d)


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="83d3a980")
def oracle_forward(inputs):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
        *_shape_params,
    ) = inputs
    device = arg0.device

    # -- Kernel 1: sum arg0+arg1 into a contiguous bf16 buffer.
    total = N * C40 * H * W  # 16056320
    added_contig = torch.empty(total, device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    BLOCK_ADD = 1024  # divides total
    ct.launch(
        stream,
        (total // BLOCK_ADD, 1, 1),
        _add_bf16_kernel,
        (arg0.reshape(total), arg1.reshape(total), added_contig, BLOCK_ADD),
    )
    added_contig = added_contig.view(N, C40, H, W)

    # Materialize copy_1 in channels-last layout.
    copy_1 = torch.empty_strided(
        (N, C40, H, W),
        (C40 * HW, 1, W * C40, C40),
        device=device,
        dtype=torch.bfloat16,
    )
    copy_1.copy_(added_contig)

    # -- Kernel 2 + 3: channel reductions (sum_1..sum_4, mul_8, mul_17)
    # via partial+finalize in cuTile.
    # View added_contig as [N*HW, C40] channels-last so C40 is contiguous.
    added_nhc = added_contig.permute(0, 2, 3, 1).contiguous().view(N * HW, C40)
    added_slice_nhc = added_nhc[:, C20:C40].contiguous()  # [N*HW, C20]
    arg2_nhc = arg2.permute(0, 2, 3, 1).contiguous().view(N * HW, C40)
    arg6_nhc = arg6.permute(0, 2, 3, 1).contiguous().view(N * HW, C20)
    arg3_flat_f = arg3.view(C40).to(torch.float32).contiguous()
    arg7_flat_f = arg7.view(C20).to(torch.float32).contiguous()

    ROWS_TOTAL = N * HW
    C40_PAD = 64
    C20_PAD = 32

    partial_x40 = torch.empty((NUM_TILES, C40_PAD), device=device, dtype=torch.float32)
    partial_xrhs40 = torch.empty((NUM_TILES, C40_PAD), device=device, dtype=torch.float32)
    partial_x20 = torch.empty((NUM_TILES, C20_PAD), device=device, dtype=torch.float32)
    partial_xrhs20 = torch.empty((NUM_TILES, C20_PAD), device=device, dtype=torch.float32)

    ct.launch(
        stream,
        (NUM_TILES, 1, 1),
        _partial_reductions_kernel,
        (added_nhc, arg2_nhc, arg3_flat_f, added_slice_nhc, arg6_nhc, arg7_flat_f,
         partial_x40, partial_xrhs40, partial_x20, partial_xrhs20,
         ROWS_TOTAL, ROWS_PER_TILE, C40_PAD, C20_PAD, C40, C20),
    )

    sum_1 = torch.empty((C40_PAD,), device=device, dtype=torch.float32)
    mul_8 = torch.empty((C40_PAD,), device=device, dtype=torch.float32)
    sum_3 = torch.empty((C20_PAD,), device=device, dtype=torch.float32)
    mul_17 = torch.empty((C20_PAD,), device=device, dtype=torch.float32)

    tiles_pad = _next_pow2(NUM_TILES)
    arg4_padded = torch.zeros((C40_PAD,), device=device, dtype=torch.float32)
    arg4_padded[:C40].copy_(arg4)
    arg8_padded = torch.zeros((C20_PAD,), device=device, dtype=torch.float32)
    arg8_padded[:C20].copy_(arg8)

    ct.launch(
        stream,
        (1, 1, 1),
        _finalize_reduce_kernel,
        (partial_x40, partial_xrhs40, partial_x20, partial_xrhs20,
         arg4_padded, arg8_padded,
         sum_1, mul_8, sum_3, mul_17,
         NUM_TILES, tiles_pad, C40_PAD, C20_PAD, C40, C20),
    )
    sum_1 = sum_1[:C40]
    mul_8 = mul_8[:C40]
    sum_3 = sum_3[:C20]
    mul_17 = mul_17[:C20]

    # Recover sum_2, sum_4 from the mul_8, mul_17 relations. Actually we need
    # sum_2 for coef_a40 and sum_4 for coef_a20. Compute them via a small torch
    # divide (arg4, arg8 are per-channel f32 tensors).
    # Actually simpler: recompute in torch from partial tensors — cheap 2D reduce.
    # Better: we already have sum_2 = mul_8 / arg4 arithmetically; do that.
    sum_2 = mul_8 / arg4  # [C40] — well-defined since arg4 is non-zero scale
    sum_4 = mul_17 / arg8  # [C20]

    # Coefficients (torch elementwise ops on small tensors)
    coef_a40 = sum_2 * INV_ROWS * arg4 * arg4  # [40]
    coef_b40 = sum_1 * INV_ROWS  # [40]
    coef_c40 = arg4 * arg5  # [40]
    coef_a20 = sum_4 * INV_ROWS * arg8 * arg8  # [20]
    coef_b20 = sum_3 * INV_ROWS  # [20]
    coef_c20 = arg8 * arg9  # [20]

    arg3_flat = arg3.reshape(C40).contiguous()
    arg7_flat = arg7.reshape(C20).contiguous()

    # -- Kernel 4: grad40 output (bf16 contiguous [N, C40, H, W])
    ce_2 = torch.empty((N, C40, H, W), device=device, dtype=torch.bfloat16)
    copy_1_2d = added_contig.reshape(N * C40, HW)
    arg2_2d = arg2.reshape(N * C40, HW)
    ce_2_2d = ce_2.view(N * C40, HW)
    ct.launch(
        stream,
        (N * C40, 1, 1),
        _grad_row_kernel,
        (copy_1_2d, arg2_2d, arg3_flat, coef_a40, coef_b40, coef_c40,
         ce_2_2d, HW, HW_PAD, C40),
    )

    # -- Kernel 5: grad20 output (bf16 channels-last [N, C20, H, W])
    ce_5 = torch.empty_strided(
        (N, C20, H, W),
        (C20 * HW, 1, W * C20, C20),
        device=device,
        dtype=torch.bfloat16,
    )
    slice_contig = added_contig[:, C20:C40].contiguous()  # bf16 [N, 20, H, W] contig
    slice_2d = slice_contig.reshape(N * C20, HW)
    arg6_2d = arg6.reshape(N * C20, HW)
    ce_5_contig = torch.empty((N, C20, H, W), device=device, dtype=torch.bfloat16)
    ce_5_contig_2d = ce_5_contig.view(N * C20, HW)
    ct.launch(
        stream,
        (N * C20, 1, 1),
        _grad_row_kernel,
        (slice_2d, arg6_2d.contiguous(), arg7_flat,
         coef_a20, coef_b20, coef_c20,
         ce_5_contig_2d, HW, HW_PAD, C20),
    )
    ce_5.copy_(ce_5_contig)

    return copy_1, sum_1, mul_8, ce_2, sum_3, mul_17, ce_5
