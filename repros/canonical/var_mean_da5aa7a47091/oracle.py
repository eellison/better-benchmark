"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete residual-add LayerNorm scope, including correction=0 row statistics, eps=1e-12 rsqrt, affine epilogue, bf16 output cast, and returned reshape alias, in one point-specialized Triton row kernel, whereas Inductor schedules the norm-template reduction and aliasing output scope through its generic fused normalization path; Inductor cannot do this today because its scheduler does not have a hidden-dimension LayerNorm template that preserves this full residual-add and returned-view contract as one scheduled unit; the fix is SCHEDULER_FUSION: teach norm-template fusion to emit the complete residual-add LayerNorm with direct alias-view output handling."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _residual_layernorm_view_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    ROUND_ADD: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask
    offsets = rows * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    x = flat + residual
    if ROUND_ADD:
        x = x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=1) / HIDDEN
        centered = x - mean[:, None]
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / HIDDEN
    else:
        mean_acc = tl.zeros([ROW_BLOCK, BLOCK_N], tl.float32)
        m2_acc = tl.zeros([ROW_BLOCK, BLOCK_N], tl.float32)
        weight_acc = tl.zeros([ROW_BLOCK, BLOCK_N], tl.float32)
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            x,
            mean_acc,
            m2_acc,
            weight_acc,
            True,
        )
        mean_acc = tl.where(mask, mean_next, mean_acc)
        m2_acc = tl.where(mask, m2_next, m2_acc)
        weight_acc = tl.where(mask, weight_next, weight_acc)
        mean, m2, _weight = triton_helpers.welford(
            mean_acc,
            m2_acc,
            weight_acc,
            1,
        )
        centered = x - mean[:, None]
        variance = m2 / HIDDEN
    y = (
        centered * libdevice.rsqrt(variance[:, None] + 1.0e-12) * weight + bias
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(norm_out_ptr + offsets, y, mask=mask)

@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024, ROW_BLOCK=2, ROUND_ADD=True, num_warps=4)
@oracle_impl(hardware="B200", point="107053b2", BLOCK_N=2048, ROW_BLOCK=1, ROUND_ADD=False, num_warps=8)
@oracle_impl(hardware="B200", point="82b81ff3", BLOCK_N=1024, ROW_BLOCK=2, ROUND_ADD=True, num_warps=4)
@oracle_impl(hardware="B200", point="94a8a62c", BLOCK_N=256, ROW_BLOCK=4, ROUND_ADD=True, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N, ROW_BLOCK, ROUND_ADD, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    norm_shape = tuple(int(dim) for dim in _shape_param_0)
    view_shape = tuple(int(dim) for dim in _shape_param_1)

    norm_out = torch.empty_strided(
        norm_shape,
        (norm_shape[1] * norm_shape[2], norm_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _residual_layernorm_view_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        norm_out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_N=BLOCK_N,
        ROW_BLOCK=ROW_BLOCK,
        ROUND_ADD=ROUND_ADD,
        num_warps=num_warps,
    )
    return (norm_out, norm_out.reshape(view_shape))
