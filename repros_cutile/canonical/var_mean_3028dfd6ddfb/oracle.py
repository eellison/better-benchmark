"""cuTile port of var_mean_3028dfd6ddfb: MobileViT patch-unfold LayerNorm.

Fused: gather from the channels-last NCHW input directly using computed
strided offsets (avoids all the torch-side .contiguous() copies).  Then
write the pre-norm layout, mean, rsqrt, and affine LN output in one kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@ct.kernel
def _mobilevit_patch_layernorm_kernel(
    x_ptr,           # bf16 flat view of arg0 [N*C*H*W total]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    layout_ptr,      # bf16 flat [ROWS * HIDDEN]
    mean_ptr,        # f32 [ROWS]
    rsqrt_ptr,       # f32 [ROWS]
    out_ptr,         # bf16 flat [ROWS * HIDDEN]
    stride_n: ct.Constant[int],
    stride_c: ct.Constant[int],
    stride_h: ct.Constant[int],
    stride_w: ct.Constant[int],
    ROWS_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    WIDTH_C: ct.Constant[int],
    PATCHES_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    block = ct.bid(0)
    rows_1d = block * ROW_BLOCK + ct.arange(ROW_BLOCK, dtype=ct.int32)  # [RB]

    patch = rows_1d - (rows_1d // PATCHES_C) * PATCHES_C
    row_group = rows_1d // PATCHES_C
    lane = row_group - (row_group // 4) * 4
    batch = row_group // 4
    half_width = WIDTH_C // 2
    patch_h = patch // half_width
    patch_w = patch - patch_h * half_width
    h = patch_h * 2 + lane // 2
    w = patch_w * 2 + lane - (lane // 2) * 2

    cols_1d = ct.arange(BLOCK_H, dtype=ct.int32)
    rows_2d = ct.reshape(rows_1d, (ROW_BLOCK, 1))
    cols_2d = ct.reshape(cols_1d, (1, BLOCK_H))
    rows_bc = ct.broadcast_to(rows_2d, (ROW_BLOCK, BLOCK_H))
    cols_bc = ct.broadcast_to(cols_2d, (ROW_BLOCK, BLOCK_H))

    batch_2d = ct.broadcast_to(ct.reshape(batch, (ROW_BLOCK, 1)),
                               (ROW_BLOCK, BLOCK_H))
    h_2d = ct.broadcast_to(ct.reshape(h, (ROW_BLOCK, 1)),
                           (ROW_BLOCK, BLOCK_H))
    w_2d = ct.broadcast_to(ct.reshape(w, (ROW_BLOCK, 1)),
                           (ROW_BLOCK, BLOCK_H))

    input_offsets = batch_2d * stride_n + cols_bc * stride_c + h_2d * stride_h + w_2d * stride_w
    col_valid = cols_bc < HIDDEN_C
    # Clamp OOB to 0 to keep gather in-bounds (masked out later).
    safe_offsets = ct.where(col_valid, input_offsets,
                            ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.int32))
    x_bf16 = ct.gather(x_ptr, (safe_offsets,))
    values = ct.astype(x_bf16, ct.float32)

    # Write pre-norm layout to layout_ptr at rows_bc * HIDDEN + cols_bc.
    output_offsets = rows_bc * HIDDEN_C + cols_bc
    ct.scatter(layout_ptr, (output_offsets,), x_bf16, mask=col_valid)

    zero = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)
    values_masked = ct.where(col_valid, values, zero)
    sum_values = ct.sum(values_masked, axis=1, keepdims=True)
    mean = sum_values * (1.0 / HIDDEN_F)
    centered = values - mean
    centered_masked = ct.where(col_valid, centered, zero)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / HIDDEN_F)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)

    # Store mean, rsqrt, and affine.
    mean_1d = ct.reshape(mean, (ROW_BLOCK,))
    invstd_1d = ct.reshape(invstd, (ROW_BLOCK,))
    ct.store(mean_ptr, index=(block,), tile=mean_1d)
    ct.store(rsqrt_ptr, index=(block,), tile=invstd_1d)
    ct.scatter(out_ptr, (output_offsets,), affine_bf, mask=col_valid)


@oracle_impl(hardware="B200", point="11d32ae7", BLOCK_H=256, ROW_BLOCK=4)
@oracle_impl(hardware="B200", point="354f721b", BLOCK_H=256, ROW_BLOCK=4)
@oracle_impl(hardware="B200", point="3fce1def", BLOCK_H=256, ROW_BLOCK=4)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, _shape0, _shape1, _shape2, _shape3 = inputs
    layout_shape = _as_shape(_shape2)
    out_shape = _as_shape(_shape3)
    hidden = int(layout_shape[2])
    patches = int(layout_shape[1])
    rows = int(out_shape[0])
    stat_shape = (int(layout_shape[0]), patches, 1)

    layout = torch.empty_strided(
        layout_shape,
        _contiguous_stride(layout_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Flat views so cuTile sees them as 1D arrays.
    n, c, h, w = arg0_1.shape
    x_flat_len = int(n * c * h * w)  # total elements in memory
    # arg0_1 is channels-last with stride (C*H*W, 1, W*C, C); total memory
    # spans N*C*H*W elements.
    x_1d = torch.as_strided(arg0_1, (x_flat_len,), (1,))
    layout_1d = layout.view(-1)
    mean_1d = mean.view(-1)
    rsqrt_1d = rsqrt.view(-1)
    out_1d = out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _mobilevit_patch_layernorm_kernel,
        (x_1d, arg1_1, arg2_1, layout_1d, mean_1d, rsqrt_1d, out_1d,
         int(arg0_1.stride(0)), int(arg0_1.stride(1)),
         int(arg0_1.stride(2)), int(arg0_1.stride(3)),
         rows, hidden, int(arg0_1.shape[3]), patches,
         BLOCK_H, ROW_BLOCK, float(hidden)),
    )
    return layout, mean, rsqrt, out
