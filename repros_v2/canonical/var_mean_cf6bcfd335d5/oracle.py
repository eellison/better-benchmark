"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Longformer bf16 residual-add LayerNorm and three separate contiguous clone returns in one row kernel, preserving the bf16 residual-add rounding boundary before f32 statistics, centered population variance order, eps=1e-5 before libdevice rsqrt, f32 affine order, final bf16 output cast, and sequence-major clone layouts, whereas Inductor currently emits the normalization output and then a separate clone/layout-copy kernel for the three returned sequence-major views; Inductor cannot do this today because the scheduler does not fuse a fixed-hidden row-normalization lowering with multiple required transposed clone stores in one full-scope multi-output epilogue; the fix is SCHEDULER_FUSION: teach the normalization schedule to sink compatible returned clone materializations into the row kernel while preserving the visible contiguous output and clone storage contracts."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 8192
BATCH = 8
SEQ = 1024
HIDDEN = 768
BLOCK_H = 1024
EPS = 1.0e-5


@triton.jit
def _bf16_residual_layernorm_clones_kernel(
    addmm_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    ROW_BLOCK: tl.constexpr,
    ROWS_: tl.constexpr,
    BATCH_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_H_: tl.constexpr,
    EPS_: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H_)[None, :]
    mask = (rows < ROWS_) & (cols < HIDDEN_)
    offsets = rows * HIDDEN_ + cols

    addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    added = (addmm + residual).to(tl.bfloat16, fp_downcast_rounding="rtne").to(
        tl.float32
    )

    reduce_values = tl.where(mask, added, 0.0)
    mean = tl.sum(reduce_values, axis=1)[:, None].to(tl.float32) / tl.full(
        (1, 1), HIDDEN_, tl.float32
    )
    centered = added - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[
        :, None
    ].to(tl.float32) / tl.full((1, 1), HIDDEN_, tl.float32)
    invstd = libdevice.rsqrt(variance + tl.full((1, 1), EPS_, tl.float32))

    weight = tl.load(weight_ptr + cols, mask=cols < HIDDEN_, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cols < HIDDEN_, other=0.0).to(tl.float32)
    out = centered * invstd
    out = out * weight + bias

    tl.store(out0_ptr + offsets, out, mask=mask)

    flat_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    seq = flat_rows - (flat_rows // SEQ_) * SEQ_
    batch = flat_rows // SEQ_
    clone_offsets = (seq * BATCH_ + batch) * HIDDEN_ + cols
    tl.store(out1_ptr + clone_offsets, out, mask=mask)
    tl.store(out2_ptr + clone_offsets, out, mask=mask)
    tl.store(out3_ptr + clone_offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="1cea4d76", ROW_BLOCK=1, num_warps=8)
def oracle_forward(inputs, *, ROW_BLOCK: int, num_warps: int):
    addmm, residual, weight, bias, _shape0, shape1, shape2, shape3 = inputs
    out0 = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.bfloat16,
    )
    out1 = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.bfloat16,
    )
    out2 = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.bfloat16,
    )
    out3 = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.bfloat16,
    )
    _bf16_residual_layernorm_clones_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        addmm,
        residual,
        weight,
        bias,
        out0,
        out1,
        out2,
        out3,
        ROW_BLOCK=ROW_BLOCK,
        ROWS_=ROWS,
        BATCH_=BATCH,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_H_=BLOCK_H,
        EPS_=EPS,
        num_warps=num_warps,
    )
    return out0, out1.view(tuple(shape1)), out2.view(tuple(shape2)), out3.view(tuple(shape3))
