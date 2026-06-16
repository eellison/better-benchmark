"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeepRecommender bf16 dropout-gated SELU-style producer once, including the compiled-path bool scale-by-5 returned bf16 activation, transpose view, and sibling column sum with the captured rounded-product and final bf16-to-fp32 sum cast in one Triton kernel, whereas Inductor schedules the gated pointwise producer, materialized output, transpose alias, and reduction consumer as generic work over the same expression; Inductor cannot do this today because its scheduler does not fuse a required materialized side output and compatible sibling reduction while preserving the observable bf16 output and reduction rounding boundaries; the fix is SCHEDULER_FUSION: teach scheduler/codegen to emit a multi-output pointwise-plus-column-reduction plan for shared bf16 producers with alias returns and dtype-cast reduction epilogues."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 256
COLS = 1024


@triton.jit
def _gated_selu_bf16_sum_kernel(
    keep_ptr,
    x_ptr,
    gate_ptr,
    out_ptr,
    sum_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_start = tl.program_id(0) * BLOCK_N
    rows = tl.arange(0, ROWS_)[:, None]
    cols = col_start + tl.arange(0, BLOCK_N)[None, :]
    valid = cols < COLS_
    offsets = rows * COLS_ + cols

    keep = tl.load(keep_ptr + offsets, mask=valid, other=0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    keep_scale = (keep * 5.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    x_scaled = x * keep_scale.to(tl.float32)

    exp_gate = libdevice.exp(gate * 1.0)
    negative = (x_scaled * 1.0) * 1.7580993408473766
    negative = negative * exp_gate
    positive = x_scaled * 1.0507009873554805
    value = tl.where(gate <= 0.0, negative, positive)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets, value_bf16, mask=valid)

    x_scaled_sum = (x * keep_scale.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    negative_sum = (x_scaled_sum * 1.0) * 1.7580993408473766
    negative_sum = negative_sum * exp_gate
    positive_sum = x_scaled_sum * 1.0507009873554805
    value_sum = tl.where(gate <= 0.0, negative_sum, positive_sum)
    value_sum_bf16 = value_sum.to(tl.bfloat16, fp_downcast_rounding="rtne")
    col_sum = tl.sum(tl.where(valid, value_sum_bf16.to(tl.float32), 0.0), axis=0)
    rounded_sum = col_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    sum_cols = col_start + tl.arange(0, BLOCK_N)
    tl.store(sum_ptr + sum_cols, rounded_sum, mask=sum_cols < COLS_)


# 35648b40: (T([256,1024], b8), T([256,1024], bf16), T([256,1024], bf16), S([1024]))
@oracle_impl(hardware="B200", point="35648b40", BLOCK_N=2, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    keep, x, gate, shape_param = inputs

    base = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    _gated_selu_bf16_sum_kernel[(triton.cdiv(COLS, BLOCK_N),)](
        keep,
        x,
        gate,
        base,
        sum_out,
        ROWS_=ROWS,
        COLS_=COLS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return base, base.permute(1, 0), sum_out
