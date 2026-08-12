"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete VisFormer dense channels-last residual add as one storage-linear Triton pointwise kernel, including the f32 input, bf16 input promotion to f32, final bf16 rounding, and fresh channels-last bf16 output layout for both captured shapes, whereas Inductor already lowers this isolated add/cast scope to the same mandatory two-read/one-write memory traffic envelope; Inductor cannot materially improve this repro through local scheduler fusion, algebraic elimination, or a new pattern because there is no adjacent producer or consumer to fuse and the remaining cost is dominated by bandwidth plus launch overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor channels-last pointwise case unless broader pointwise bandwidth or launch-overhead work moves both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_cast_bf16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


@oracle_impl(hardware="B200", point="ac403ece")
@oracle_impl(hardware="B200", point="509debd6")
def oracle_forward(inputs):
    arg0_1, arg1_1 = inputs
    output = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_elements = arg0_1.numel()
    grid = lambda meta: (n_elements // meta["BLOCK_SIZE"],)
    _add_cast_bf16_kernel[grid](
        arg0_1,
        arg1_1,
        output,
        n_elements=n_elements,
    )
    return output
