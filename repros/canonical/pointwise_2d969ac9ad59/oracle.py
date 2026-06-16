"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full bf16/f32 tanh-approximate GELU scope, including the metadata-only `[M, N] -> [B, S, N] -> [M, N]` views, Inductor-style fp32 fused `0.5*x*(1+tanh(0.7978845608028654*(x+0.044715*x*x*x)))` arithmetic, final dtype-preserving store, and contiguous output view, in one storage-linear Triton kernel registered at every captured shape point, whereas Inductor lowers the same view-plus-GELU chain through its generic pointwise fusion path; Inductor cannot do this today because the scheduler has no dedicated B200-tuned tanh-GELU pointwise template for this cross-model shape family and must rely on generic pointwise launch choices; the fix is NEW_PATTERN: add a guarded tanh-approximate GELU materialization template or equivalent pointwise autotuning specialization for metadata-view transformer GELU patterns."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _tanh_gelu_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(input_ptr + offsets).to(tl.float32)
    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    y = (x * 0.5) * (libdevice.tanh(tanh_arg) + 1.0)
    tl.store(output_ptr + offsets, y)


def _resolve_shape(shape, numel):
    out = [int(dim) for dim in shape]
    infer = -1
    known = 1
    for index, dim in enumerate(out):
        if dim == -1:
            infer = index
        else:
            known *= dim
    if infer >= 0:
        out[infer] = numel // known
    return tuple(out)


@oracle_impl(hardware="B200", point="cd47785e", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="92efc45e", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="d05618d1", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="7b14189f", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5cd82a46", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="8f687505", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    x, _shape0, shape1 = inputs
    out_shape = _resolve_shape(shape1, x.numel())
    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=x.dtype,
    )
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    _tanh_gelu_kernel[grid](
        x,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
