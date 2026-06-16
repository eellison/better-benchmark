"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 transformer tanh-approximate GELU scope, including the metadata-only `[M, N] -> [B, S, N] -> [M, N]` views, the explicit bf16-to-fp32 cast for the cubic/tanh path, the bf16-rounded `x * 0.5` factor, final bf16 store, and contiguous output view, in one storage-linear Triton kernel registered at every captured shape point, whereas Inductor lowers the same view/cast/GELU/cast chain through generic pointwise fusion; Inductor cannot do this today because pointwise codegen has no guarded B200-tuned tanh-GELU template that preserves this mixed bf16/fp32 rounding structure while specializing the launch to the transformer hidden-size family; the fix is NEW_PATTERN: add a dedicated metadata-view tanh-GELU pointwise template or equivalent autotuned specialization that keeps the explicit cast boundaries and uses the natural tanh lowering."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bf16_fp32_tanh_gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    half = (x * 0.5).to(tl.bfloat16).to(tl.float32)
    x_cubed = x * x * x
    tanh_arg = (x + x_cubed * 0.044715) * 0.7978845608028654
    y = half * (libdevice.tanh(tanh_arg) + 1.0)
    tl.store(output_ptr + offsets, y, mask=mask)


def _resolve_shape(shape, numel):
    out = [int(dim) for dim in shape]
    infer_index = -1
    known = 1
    for index, dim in enumerate(out):
        if dim == -1:
            infer_index = index
        else:
            known *= dim
    if infer_index >= 0:
        out[infer_index] = numel // known
    return tuple(out)


# 7b14189f: (T([128, 16384], bf16), S([1, 128, 16384]), S([128, 16384]))
# cd47785e: (T([4096, 16384], bf16), S([8, 512, 16384]), S([4096, 16384]))
# 5cd82a46: (T([4096, 8192], bf16), S([32, 128, 8192]), S([4096, 8192]))
@oracle_impl(hardware="B200", point="7b14189f", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="cd47785e", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5cd82a46", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    x, _shape0, shape1 = inputs
    out_shape = _resolve_shape(shape1, x.numel())
    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    _bf16_fp32_tanh_gelu_kernel[grid](
        x,
        output,
        n_elements=x.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
