"""cuTile port of var_mean_51bba8502cc5: MobileViT patch unfold + LayerNorm.

Fuses the patch-unfold materialization with masked LayerNorm over hidden in
ONE cuTile kernel, mirroring Triton's single-kernel structure and
BLOCK_H=256, ROW_BLOCK=4. Uses padding_mode=ZERO for weight/bias tile and
mask-guarded scatter for the non-pow2 hidden dim. Avoids torch
`.contiguous()` clones and `torch.zeros(padded_shape)` copies entirely by
indexing the strided channels-last source directly via ct.gather.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _mobilevit_patch_layernorm_kernel(
    x_ptr,           # bf16 flat 1D view of channels-last [N, C, H, W]
    weight_ptr,      # bf16 [HIDDEN]
    bias_ptr,        # bf16 [HIDDEN]
    layout_ptr,      # bf16 flat 1D view of [ROWS, HIDDEN]
    out_ptr,         # bf16 flat 1D view of [ROWS, HIDDEN]
    ROWS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    WIDTH: ct.Constant[int],
    PATCHES: ct.Constant[int],
    STRIDE_N: ct.Constant[int],
    STRIDE_C: ct.Constant[int],
    STRIDE_H: ct.Constant[int],
    STRIDE_W: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_base = ct.bid(0) * ROW_BLOCK
    row_offs = ct.arange(ROW_BLOCK, dtype=ct.int32)
    col_offs = ct.arange(BLOCK_H, dtype=ct.int32)
    rows_2d = ct.reshape(row_base + row_offs, (ROW_BLOCK, 1))
    cols_2d = ct.reshape(col_offs, (1, BLOCK_H))
    zero_b = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.bool_)
    row_valid = (rows_2d < ROWS) | zero_b
    col_valid = (cols_2d < HIDDEN) | zero_b
    mask = row_valid & col_valid

    half_width = WIDTH // 2
    patch = rows_2d % PATCHES
    row_group = rows_2d // PATCHES
    lane = row_group % 4
    batch = row_group // 4
    patch_h = patch // half_width
    patch_w = patch - patch_h * half_width
    h = patch_h * 2 + lane // 2
    w = patch_w * 2 + lane - (lane // 2) * 2

    input_offsets = (batch * STRIDE_N + cols_2d * STRIDE_C
                     + h * STRIDE_H + w * STRIDE_W)
    output_offsets = rows_2d * HIDDEN + cols_2d

    x_bf = ct.gather(x_ptr, input_offsets, mask=mask, padding_value=0.0)
    ct.scatter(layout_ptr, output_offsets, x_bf, mask=mask)
    values = ct.astype(x_bf, ct.float32)

    values_masked = ct.where(mask, values, 0.0)
    sum_values = ct.sum(values_masked, axis=1, keepdims=True)
    mean = sum_values / HIDDEN
    centered = values - mean
    variance = (ct.sum(ct.where(mask, centered * centered, 0.0),
                       axis=1, keepdims=True) / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.reshape(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO),
        (1, BLOCK_H))
    bias = ct.reshape(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO),
        (1, BLOCK_H))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    y = centered * invstd * weight_f + bias_f
    ct.scatter(out_ptr, output_offsets,
               ct.astype(y, ct.bfloat16), mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="5801101d", BLOCK_H=256, ROW_BLOCK=4)
@oracle_impl(hardware="B200", point="6d2833c8", BLOCK_H=256, ROW_BLOCK=4)
@oracle_impl(hardware="B200", point="58fa573e", BLOCK_H=256, ROW_BLOCK=4)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, s0, s1, s2, s3 = inputs
    layout_shape = _as_shape(s2)      # [512, patches, hidden]
    out_shape = _as_shape(s3)         # [rows, hidden]
    hidden = int(layout_shape[2])
    patches = int(layout_shape[1])
    rows = int(out_shape[0])
    width = int(arg0_1.shape[3])

    layout = torch.empty_strided(
        layout_shape,
        (patches * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Metadata-only flat views (no copies): input storage is contiguous NHWC
    # under channels-last strides; layout/out are already contiguous.
    x_flat = torch.as_strided(arg0_1, (arg0_1.numel(),), (1,))
    layout_flat = layout.view(-1)
    out_flat = out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _mobilevit_patch_layernorm_kernel,
        (x_flat, arg1_1, arg2_1, layout_flat, out_flat,
         rows, hidden, width, patches,
         arg0_1.stride(0), arg0_1.stride(1),
         arg0_1.stride(2), arg0_1.stride(3),
         BLOCK_H, ROW_BLOCK),
    )
    return layout, out
