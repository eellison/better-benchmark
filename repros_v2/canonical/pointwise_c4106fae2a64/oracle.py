"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet bf16 ReLU plus flatten scope in one storage-linear Triton kernel, including NaN-preserving `relu(arg0)` and the fresh contiguous `[512, 1280]` output produced by the view. Inductor already lowers this isolated pointwise/view graph to equivalent dense one-read/one-write GPU work, so there is no scheduler, reduction, or layout trick left inside this scope that can materially reduce memory traffic or launch overhead. The fix is BANDWIDTH_BOUND: treat this as a pointwise memory/launch floor unless surrounding graph fusion becomes available."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 512
COLS = 1280
NUMEL = ROWS * COLS
OUT_SHAPE = (ROWS, COLS)
OUT_STRIDE = (COLS, 1)


@triton.jit
def _relu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x != x, x, tl.maximum(x, 0.0))
    tl.store(output_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="21720c2b", BLOCK=1024, num_warps=1)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, shape = inputs
    del shape

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=torch.bfloat16,
    )
    _relu_flatten_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        x,
        out,
        n_elements=NUMEL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
