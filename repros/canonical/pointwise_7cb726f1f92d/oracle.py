"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete f32-to-bf16 conversion followed by `full(shape_param, 0, dtype=bf16)` and one-dimensional cat in one segment-aware Triton kernel for each captured point, copying the input prefix with explicit RTNE bf16 rounding and storing the constant-zero suffix directly, whereas Inductor lowers the decomposed convert/full/cat graph through generic pointwise cat materialization over the concatenated index space; Inductor cannot do this today because cat lowering has no fixed-segment template that sinks a dtype-conversion producer and a tiny constant suffix into one direct materialization; the fix is NEW_PATTERN: add a guarded cat-with-cast-and-constant-tail lowering for dense one-dimensional cats."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _cast_cat_zero_tail_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    IN_N: tl.constexpr,
    OUT_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < OUT_N
    input_mask = offsets < IN_N
    values = tl.load(
        input_ptr + offsets * input_stride0,
        mask=input_mask,
        other=0.0,
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(output_ptr + offsets, values, mask=output_mask)


# c2db0b92: (T([20005], f32), S([3]))
@oracle_impl(hardware="B200", point="c2db0b92", BLOCK=1024, num_warps=4)
# 2e98d974: (T([128100], f32), S([4]))
@oracle_impl(hardware="B200", point="2e98d974", BLOCK=1024, num_warps=4)
# 6925c38d: (T([30522], f32), S([6]))
@oracle_impl(hardware="B200", point="6925c38d", BLOCK=1024, num_warps=4)
# 85cac153: (T([2], f32), S([6]))
@oracle_impl(hardware="B200", point="85cac153", BLOCK=8, num_warps=1)
# 06af09d8: (T([50265], f32), S([7]))
@oracle_impl(hardware="B200", point="06af09d8", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, shape_param_0 = inputs
    in_n = arg0_1.numel()
    tail_n = int(shape_param_0[0])
    out_n = in_n + tail_n
    out = torch.empty_strided(
        (out_n,),
        (1,),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(out_n, BLOCK),)
    _cast_cat_zero_tail_kernel[grid](
        arg0_1,
        out,
        arg0_1.stride(0),
        IN_N=in_n,
        OUT_N=out_n,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
