"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MobileBERT bf16 affine scope as one storage-linear Triton kernel over the final `[32768,128]` layout, folding both metadata-only views while preserving the fused f32 affine arithmetic and final bf16 cast boundary, whereas Inductor still schedules the captured reshape/mul/add/cast/reshape graph through its generic pointwise path for the logical view chain; Inductor cannot do this today because output planning does not canonicalize the two reshapes away before choosing the pointwise schedule and allocation shape; the fix is ALGEBRAIC_ELIMINATION: canonicalize metadata-only reshape pairs around pointwise broadcasts so codegen emits direct storage-linear indexing into the final dense layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _affine_bf16_kernel(
    input_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = offsets % 128
    x = tl.load(input_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + cols, eviction_policy="evict_last")
    bias = tl.load(bias_ptr + cols, eviction_policy="evict_last")
    y = (x * scale + bias).to(tl.bfloat16)
    tl.store(output_ptr + offsets, y)


@oracle_impl(
    hardware="B200",
    point="1398f333",
    BLOCK=2048,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    final_shape = tuple(int(dim) for dim in _shape_param_1)
    output = torch.empty_strided(
        final_shape,
        (final_shape[1], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _affine_bf16_kernel[(triton.cdiv(arg0_1.numel(), BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        output,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return output
