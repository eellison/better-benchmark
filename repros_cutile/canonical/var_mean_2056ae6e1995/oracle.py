"""cuTile port of var_mean_2056ae6e1995: GhostNet BN-train + ReLU + channel-cat.

Three cuTile kernels mirror the Triton structure:
  1. `_bn_partial_stats_kernel`: per-(channel, chunk) fp32 sum & sum^2 of input.
  2. `_bn_finalize_stats_kernel`: fold chunk partials -> mean/invstd; update
     running_mean/running_var in-place.
  3. `_bn_relu_cat_kernel`: bf16 output tile combining (skip, bn+relu) along the
     channel-doubled dim; channels-last strided output.

Reductions live inside cuTile primitives (ct.sum) — no torch.var_mean/.copy_
outside these kernels except the trivial writes back through the output tuple.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
BN_MOMENTUM = 0.1
VAR_CORRECTION = 1.0000398612827361


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


@ct.kernel
def _bn_partial_stats_kernel(
    x_ptr,               # bf16 [rows, C]   (rows = N*H*W)
    partial_sum_ptr,     # f32 [NUM_CHUNKS, C]
    partial_sum2_ptr,    # f32 [NUM_CHUNKS, C]
    ROWS_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    e_block = ct.bid(0)
    c_block = ct.bid(1)
    row_start = e_block * BLOCK_E
    row_ids = row_start + ct.arange(BLOCK_E, dtype=ct.int32)
    row_lim = ct.full(shape=(BLOCK_E,), fill_value=ROWS_, dtype=ct.int32)
    row_valid = row_ids < row_lim

    col_ids = c_block * C_BLOCK + ct.arange(C_BLOCK, dtype=ct.int32)
    col_lim = ct.full(shape=(C_BLOCK,), fill_value=C_, dtype=ct.int32)
    col_valid = col_ids < col_lim

    vals_bf = ct.load(
        x_ptr, index=(e_block, c_block), shape=(BLOCK_E, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    vals = ct.astype(vals_bf, ct.float32)
    row_valid_2d = ct.reshape(row_valid, (BLOCK_E, 1))
    col_valid_2d = ct.reshape(col_valid, (1, C_BLOCK))
    valid = row_valid_2d & col_valid_2d
    zeros = ct.zeros((BLOCK_E, C_BLOCK), dtype=ct.float32)
    vals_masked = ct.where(valid, vals, zeros)
    sums = ct.sum(vals_masked, axis=0, keepdims=False)          # (C_BLOCK,)
    sums2 = ct.sum(vals_masked * vals_masked, axis=0, keepdims=False)

    ct.store(partial_sum_ptr, index=(e_block, c_block),
             tile=ct.reshape(sums, (1, C_BLOCK)))
    ct.store(partial_sum2_ptr, index=(e_block, c_block),
             tile=ct.reshape(sums2, (1, C_BLOCK)))


@ct.kernel
def _bn_finalize_stats_kernel(
    partial_sum_ptr,     # f32 [NUM_CHUNKS, C]
    partial_sum2_ptr,    # f32 [NUM_CHUNKS, C]
    running_mean_ptr,    # f32 [C] (updated in place)
    running_var_ptr,     # f32 [C] (updated in place)
    mean_ptr,            # f32 [C]
    invstd_ptr,          # f32 [C]
    C_: ct.Constant[int],
    E_: ct.Constant[int],
    NUM_CHUNKS_: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
    VAR_CORRECTION_: ct.Constant[float],
):
    c_block = ct.bid(0)
    col_ids = c_block * C_BLOCK + ct.arange(C_BLOCK, dtype=ct.int32)
    col_lim = ct.full(shape=(C_BLOCK,), fill_value=C_, dtype=ct.int32)
    col_valid = col_ids < col_lim

    sums = ct.load(
        partial_sum_ptr, index=(0, c_block), shape=(BLOCK_CHUNKS, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sums2 = ct.load(
        partial_sum2_ptr, index=(0, c_block), shape=(BLOCK_CHUNKS, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_ids = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    row_lim = ct.full(shape=(BLOCK_CHUNKS,), fill_value=NUM_CHUNKS_, dtype=ct.int32)
    row_valid = row_ids < row_lim
    row_valid_2d = ct.reshape(row_valid, (BLOCK_CHUNKS, 1))
    col_valid_2d = ct.reshape(col_valid, (1, C_BLOCK))
    valid = row_valid_2d & col_valid_2d
    zeros_2d = ct.zeros((BLOCK_CHUNKS, C_BLOCK), dtype=ct.float32)
    sums_m = ct.where(valid, sums, zeros_2d)
    sums2_m = ct.where(valid, sums2, zeros_2d)
    total = ct.sum(sums_m, axis=0, keepdims=False)   # (C_BLOCK,)
    total2 = ct.sum(sums2_m, axis=0, keepdims=False)

    inv_e = 1.0 / E_
    mean = total * inv_e
    var = total2 * inv_e - mean * mean
    zeros_c = ct.zeros((C_BLOCK,), dtype=ct.float32)
    var = ct.where(var > zeros_c, var, zeros_c)
    invstd = ct.rsqrt(var + EPS)

    old_mean = ct.load(
        running_mean_ptr, index=(c_block,), shape=(C_BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    old_var = ct.load(
        running_var_ptr, index=(c_block,), shape=(C_BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    new_mean = old_mean * 0.9 + mean * BN_MOMENTUM
    new_var = old_var * 0.9 + var * VAR_CORRECTION_ * BN_MOMENTUM

    # Scatter writes: mask out c_block tail if col_valid is not all-true.
    col_offs = c_block * C_BLOCK + ct.arange(C_BLOCK, dtype=ct.int32)
    ct.scatter(running_mean_ptr, col_offs, new_mean, mask=col_valid)
    ct.scatter(running_var_ptr, col_offs, new_var, mask=col_valid)
    ct.scatter(mean_ptr, col_offs, mean, mask=col_valid)
    ct.scatter(invstd_ptr, col_offs, invstd, mask=col_valid)


@ct.kernel
def _bn_relu_cat_kernel(
    x_ptr,          # bf16 flat [rows*C] (NHWC layout of x)
    skip_ptr,       # bf16 flat [rows*C] (NHWC layout of skip)
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    out_ptr,        # bf16 flat [rows*2*C] (NHWC layout, channel-dim = 2*C)
    ROWS_: ct.Constant[int],
    C_: ct.Constant[int],
    TWO_C_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    r_block = ct.bid(0)
    r_ids = r_block * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    r_lim = ct.full(shape=(BLOCK_R,), fill_value=ROWS_, dtype=ct.int32)
    r_valid = r_ids < r_lim

    c_ids = ct.arange(C_BLOCK, dtype=ct.int32)
    c_lim = ct.full(shape=(C_BLOCK,), fill_value=C_, dtype=ct.int32)
    c_valid = c_ids < c_lim

    x_bf = ct.load(
        x_ptr, index=(r_block, 0), shape=(BLOCK_R, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    skip_bf = ct.load(
        skip_ptr, index=(r_block, 0), shape=(BLOCK_R, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean = ct.load(
        mean_ptr, index=(0,), shape=(C_BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    invstd = ct.load(
        invstd_ptr, index=(0,), shape=(C_BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight = ct.load(
        weight_ptr, index=(0,), shape=(C_BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(C_BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    x_f = ct.astype(x_bf, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_BLOCK))
    invstd_2d = ct.reshape(invstd, (1, C_BLOCK))
    weight_2d = ct.reshape(weight, (1, C_BLOCK))
    bias_2d = ct.reshape(bias, (1, C_BLOCK))
    affine = ((x_f - mean_2d) * invstd_2d) * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK_R, C_BLOCK), dtype=ct.bfloat16)
    relu_bf = ct.where(affine_bf > zero_bf, affine_bf, zero_bf)

    # Scatter to out at columns [0..C) for skip and [C..2C) for relu.
    r_ids_2d = ct.broadcast_to(ct.reshape(r_ids, (BLOCK_R, 1)),
                                (BLOCK_R, C_BLOCK))
    c_ids_2d = ct.broadcast_to(ct.reshape(c_ids, (1, C_BLOCK)),
                                (BLOCK_R, C_BLOCK))
    r_valid_2d = ct.reshape(r_valid, (BLOCK_R, 1))
    c_valid_2d = ct.reshape(c_valid, (1, C_BLOCK))
    tile_valid = r_valid_2d & c_valid_2d

    skip_off = r_ids_2d * TWO_C_ + c_ids_2d
    relu_off = r_ids_2d * TWO_C_ + c_ids_2d + C_
    ct.scatter(out_ptr, skip_off, skip_bf, mask=tile_valid)
    ct.scatter(out_ptr, relu_off, relu_bf, mask=tile_valid)


@oracle_impl(hardware="B200", point="d3838873", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=8, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="c9e11f4c", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=8, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="02e128f4", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=4, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="2fd52f04", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=4, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="139bf125", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=2, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="de73d9ef", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=1, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="023ff9dc", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=1, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="8f9cb8b8", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=1, OUT_BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_E, C_BLOCK, FINAL_C_BLOCK, OUT_BLOCK):
    x, running_mean, running_var, weight, bias, skip = inputs
    device = x.device
    n, c, h, w = (int(d) for d in x.shape)
    rows = n * h * w

    # Reshape channels-last input to (rows, C) contiguous view.
    x_nhwc = x.permute(0, 2, 3, 1).contiguous().view(rows, c)
    skip_nhwc = skip.permute(0, 2, 3, 1).contiguous().view(rows, c)

    out = torch.empty_strided(
        (n, c * 2, h, w),
        (c * 2 * h * w, 1, c * 2 * w, c * 2),
        device=device, dtype=torch.bfloat16,
    )
    # NHWC view of out (skip at cols[:C], relu at cols[C:2C]).
    out_flat = out.permute(0, 2, 3, 1).contiguous().view(-1)
    # NOTE: We cannot use .contiguous() here since out has channels-last strides;
    # instead, view the underlying storage as (rows, 2C) directly.
    # For channels-last, the memory layout is exactly (n, h, w, 2*c) contiguous,
    # which matches out.permute(0,2,3,1) naturally.
    out_2d = out.permute(0, 2, 3, 1).view(rows, c * 2)
    out_flat = out_2d.view(-1)

    mean = torch.empty((c,), device=device, dtype=torch.float32)
    invstd = torch.empty((c,), device=device, dtype=torch.float32)

    num_chunks = (rows + BLOCK_E - 1) // BLOCK_E
    block_chunks = _next_pow2(num_chunks)
    partial_sum = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_chunks, c), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_chunks, (c + C_BLOCK - 1) // C_BLOCK, 1),
        _bn_partial_stats_kernel,
        (x_nhwc, partial_sum, partial_sum2, rows, c, BLOCK_E, C_BLOCK),
    )
    ct.launch(
        stream, ((c + FINAL_C_BLOCK - 1) // FINAL_C_BLOCK, 1, 1),
        _bn_finalize_stats_kernel,
        (partial_sum, partial_sum2, running_mean, running_var, mean, invstd,
         c, rows, num_chunks, block_chunks, FINAL_C_BLOCK, VAR_CORRECTION),
    )

    # OUT_BLOCK is total elements per launch; treat as (BLOCK_R, C_BLOCK) rows.
    # For NHWC layout, we tile by rows only with C_BLOCK spanning all channels.
    OUT_C_BLOCK = _next_pow2(c)
    BLOCK_R = max(1, OUT_BLOCK // OUT_C_BLOCK)
    ct.launch(
        stream, ((rows + BLOCK_R - 1) // BLOCK_R, 1, 1),
        _bn_relu_cat_kernel,
        (x_nhwc, skip_nhwc, weight, bias, mean, invstd, out_flat,
         rows, c, c * 2, BLOCK_R, OUT_C_BLOCK),
    )

    # Return outputs matching Triton's shapes.
    mean_view = mean.view(1, c, 1, 1)
    invstd_view = invstd.view(1, c, 1, 1)
    return mean_view, invstd_view, out, running_mean, running_var
