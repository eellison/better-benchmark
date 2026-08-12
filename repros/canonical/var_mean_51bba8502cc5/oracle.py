"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileViT 2x2 patch-layout canonicalization and affine LayerNorm scope in one Triton row kernel, including the live bf16 `[512, patches, hidden]` layout output, fp32 `var_mean(correction=0, keepdim=True)` over the hidden dimension, eps=1e-5 `rsqrt`, bf16 weight/bias affine in fp32, and the final bf16 `[rows, hidden]` view store, whereas Inductor lowers the captured clone/view/permute/clone/view/permute/clone/view plus normalization graph through generic layout-copy and norm-template schedules; Inductor cannot do this today because its scheduler/codegen does not recognize this MobileViT patch-unfold LayerNorm pattern with a visible pre-normalization layout output and therefore cannot directly index the channels-last source once while emitting both returned tensors; the fix is NEW_PATTERN: add a guarded MobileViT patch-unfold LayerNorm lowering that maps rows to original NCHW logical coordinates and fuses the layout materialization with the reduction and affine bf16 epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _mobilevit_patch_layernorm_bf16_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    layout_ptr,
    out_ptr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    WIDTH: tl.constexpr,
    PATCHES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask

    patch = rows % PATCHES
    row_group = rows // PATCHES
    lane = row_group % 4
    batch = row_group // 4
    half_width = WIDTH // 2
    patch_h = patch // half_width
    patch_w = patch - patch_h * half_width
    h = patch_h * 2 + lane // 2
    w = patch_w * 2 + lane - (lane // 2) * 2

    input_offsets = batch * stride_n + cols * stride_c + h * stride_h + w * stride_w
    output_offsets = rows * HIDDEN + cols

    x_bf16 = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    values = x_bf16.to(tl.float32)
    tl.store(layout_ptr + output_offsets, x_bf16, mask=mask)

    sum_values = tl.sum(tl.where(mask, values, 0.0), axis=1)[:, None]
    mean = sum_values / HIDDEN
    centered = values - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / HIDDEN
    invstd = libdevice.rsqrt(variance + 1.0e-5)

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = (centered * invstd * weight + bias).to(tl.bfloat16)
    tl.store(out_ptr + output_offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="5801101d", BLOCK_H=256, ROW_BLOCK=4, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="6d2833c8", BLOCK_H=256, ROW_BLOCK=4, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="58fa573e", BLOCK_H=256, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    layout_shape = tuple(int(dim) for dim in _shape_param_2)
    out_shape = tuple(int(dim) for dim in _shape_param_3)
    hidden = int(layout_shape[2])
    patches = int(layout_shape[1])
    rows = int(out_shape[0])

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

    _mobilevit_patch_layernorm_bf16_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        layout,
        out,
        stride_n=arg0_1.stride(0),
        stride_c=arg0_1.stride(1),
        stride_h=arg0_1.stride(2),
        stride_w=arg0_1.stride(3),
        ROWS=rows,
        HIDDEN=hidden,
        WIDTH=int(arg0_1.shape[3]),
        PATCHES=patches,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return layout, out
