"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full `aten.ne.Scalar(arg0_1, 1)` predicate and `prims.convert_element_type.default(..., torch.int32)` materialization with one B200-tuned storage-linear Triton load/compare/cast/store kernel registered at every captured shape point, whereas Inductor uses its generic pointwise scheduler for the same tiny int64 mask materialization; Inductor cannot do this today because it has no dedicated B200 launch/config template for this cross-model integer attention-mask shape family and must rely on generic pointwise launch choices; the fix is NEW_PATTERN: add a guarded int64 scalar-ne to int32 materialization template or equivalent pointwise autotuning specialization for these dense contiguous mask tensors."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _ne_one_i64_to_i32_kernel(
    in_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(in_ptr + offsets)
    out_values = (values != 1).to(tl.int32)
    tl.store(out_ptr + offsets, out_values)


@oracle_impl(hardware="B200", point="0105f520", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="3d748156", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="26cc4258", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    (arg0_1,) = inputs
    out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.int32,
    )
    grid = (arg0_1.numel() // BLOCK,)
    _ne_one_i64_to_i32_kernel[grid](
        arg0_1,
        out,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
