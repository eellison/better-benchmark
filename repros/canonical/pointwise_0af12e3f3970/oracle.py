"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle materializes the complete contiguous prefix slice plus bf16-to-fp32 cast scope as one storage-linear Triton kernel, reading only `arg0[:-3]` and writing the fresh contiguous fp32 output for both vocabulary-sized points, while Inductor reaches the same practical one-pass prefix-slice widening-cast envelope for this full scope; Inductor cannot materially improve this local graph today because the remaining time is dominated by the required bf16 input read, fp32 output store, dtype conversion, and launch/replay overhead rather than avoidable reduction, RNG, alias, or layout work; the fix is BANDWIDTH_BOUND: record this prefix-slice cast as at floor unless broader pointwise bandwidth or launch-overhead work moves both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _prefix_slice_bf16_to_f32_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, values, mask=mask)


def _launch(inputs, *, BLOCK: int, num_warps: int):
    (arg0_1,) = inputs
    rows = int(arg0_1.shape[0]) - 3
    cols = int(arg0_1.shape[1])
    output = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    n_elements = rows * cols
    grid = (triton.cdiv(n_elements, BLOCK),)
    _prefix_slice_bf16_to_f32_kernel[grid](
        arg0_1,
        output,
        n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return output


# 9ba4aad1: (T([20008,768], bf16))
@oracle_impl(hardware="B200", point="9ba4aad1", BLOCK=1024, num_warps=4)
# 19d57796: (T([50008,768], bf16))
@oracle_impl(hardware="B200", point="19d57796", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    return _launch(inputs, BLOCK=BLOCK, num_warps=num_warps)
