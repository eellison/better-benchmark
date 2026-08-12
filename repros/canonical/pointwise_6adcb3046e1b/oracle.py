"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 MobileNetV3 hard-swish flatten scope as one storage-linear Triton pointwise pass, including bf16-to-fp32 promotion, add/clamp_min/clamp_max/mul/div arithmetic, final bf16 rounding, and the returned [512, 1280] flatten layout, whereas Inductor already lowers the captured pointwise chain and metadata view to equivalent fused global-memory traffic; Inductor cannot materially improve this repro through a local fusion rewrite because one input read, one output write, and launch overhead dominate the small arithmetic chain; the fix is BANDWIDTH_BOUND: keep this as an at-floor pointwise case unless broader pointwise codegen or launch overhead improves."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N_ELEMENTS = 512 * 1280


@triton.jit
def _hardswish_flatten_bf16_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offsets).to(tl.float32)
    shifted = x + 3.0
    clamp_min = tl.where(shifted < 0.0, 0.0, shifted)
    clamp_max = tl.where(clamp_min > 6.0, 6.0, clamp_min)
    out = (x * clamp_max) / 6.0
    tl.store(output_ptr + offsets, out.to(tl.bfloat16))


@oracle_impl(hardware="B200", point="21720c2b", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    arg0_1, _shape_param_0 = inputs
    output = torch.empty_strided(
        tuple(_shape_param_0),
        (1280, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _hardswish_flatten_bf16_kernel[(N_ELEMENTS // BLOCK_SIZE,)](
        arg0_1,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
