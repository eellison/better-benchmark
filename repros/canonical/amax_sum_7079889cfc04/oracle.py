"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete huge-row bf16 softmax forward as a Triton online softmax over each `[8192,262144]` row, streaming the f32 row max and f32 natural-exp denominator and then writing the returned bf16 probabilities without materializing full-size f32 intermediates, whereas Inductor lowers the captured cast/amax/sub/exp/sum/div/cast graph through generic wide-row reductions and pointwise materialization; Inductor cannot do this today because its softmax lowering does not specialize this extremely wide static vocabulary shape into a memory-bounded online row program that avoids f32 intermediate tensors while preserving the captured f32 math and bf16 output cast; the fix is COOPERATIVE_SPLIT_K: add a guarded wide-row online-softmax template that streams row reductions and the final bf16 epilogue for static huge-vocabulary shapes."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _online_softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * n_cols

    row_max = float("-inf")
    row_sum = 0.0
    for block_start in tl.range(0, n_cols, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    for block_start in tl.range(0, n_cols, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        out = tl.exp(x - row_max) / row_sum
        tl.store(output_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


# bf16[8192, 262144] -> bf16[8192, 262144]
@oracle_impl(hardware="B200", point="4ac62a08", BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N, num_warps):
    x = inputs[0]
    out = torch.empty_like(x)
    _online_softmax_kernel[(x.shape[0],)](
        x,
        out,
        n_cols=x.shape[1],
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
