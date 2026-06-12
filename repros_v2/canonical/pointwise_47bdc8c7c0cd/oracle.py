"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 transformer tanh-approximate GELU scope with one storage-linear Triton kernel, including the metadata-only `[M,3072] -> [B,S,3072] -> [M,3072]` views, explicit bf16-to-fp32 cubic/tanh path, the captured bf16-rounded `x * 0.5` factor, final bf16 materialization, and the returned transpose as a metadata alias of the same storage, whereas Inductor lowers the decomposed view/mul/cast/pow/mul/add/tanh/mul/view/cast/permute graph through generic pointwise and output-planning codegen; Inductor cannot do this today because pointwise codegen has no B200-tuned tanh-GELU materialization template that preserves this mixed bf16/fp32 rounding structure while exposing sibling layout aliases from one buffer; the fix is NEW_PATTERN: add a guarded bf16 tanh-GELU pointwise template with alias-aware output planning for metadata-only transpose returns."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bf16_tanh_gelu_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(input_ptr + offsets).to(tl.float32)
    half = (x * 0.5).to(tl.bfloat16).to(tl.float32)
    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    y = half * (libdevice.tanh(tanh_arg) + 1.0)
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


@oracle_impl(hardware="B200", point="92efc45e", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="d05618d1", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    x, _shape0, shape1 = inputs
    out_shape = _resolve_shape(shape1, x.numel())
    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bf16_tanh_gelu_kernel[(triton.cdiv(x.numel(), BLOCK_SIZE),)](
        x,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output, output.permute(1, 0)
