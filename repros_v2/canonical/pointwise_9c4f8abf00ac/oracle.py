"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete TrOCR finite causal-mask repro as one shape-specialized Triton fill over the dense f32[64,1,256,256] output, whereas Inductor lowers the captured iota/unsqueeze/le/expand/where chain as generic pointwise mask materialization; Inductor cannot do this today because it has no canonical lowering that recognizes a metadata-only causal-mask construction with a finite f32 minimum sentinel and emits the closed-form fill directly; the fix is NEW_PATTERN: add a guarded causal-mask pattern lowering that preserves the exact finite constants, output dtype, and output strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _finite_causal_mask_kernel(
    out,
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cols = offsets % S
    rows = (offsets // S) % S
    values = tl.where(cols <= rows, 0.0, -3.4028234663852886e38)
    tl.store(out + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch, heads, seq, _ = expand_shape
    stride = (heads * seq * seq, seq * seq, seq, 1)
    numel = batch * heads * seq * seq
    out = torch.empty_strided(
        expand_shape,
        stride,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )

    _finite_causal_mask_kernel[(triton.cdiv(numel, BLOCK),)](
        out,
        N=numel,
        S=seq,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
