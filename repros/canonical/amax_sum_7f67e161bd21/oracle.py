"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 DeBERTa attention softmax scope by eliminating the broadcast constant-false `where` mask and min-float fill, then running the stable fp32 last-dimension softmax with natural libdevice exp and storing the final contiguous bf16 `[192, 512, 512]` view, whereas Inductor lowers the decomposed full/where/amax/sub/exp/sum/div/cast/view graph through generic producers around the reduction; Inductor cannot do this today because its scheduler/codegen does not canonicalize a broadcast constant-false predicate into the identity before softmax scheduling; the fix is ALGEBRAIC_ELIMINATION: add constant-mask predicate simplification that removes the dead fill/where producer and feeds the row-softmax template directly while preserving the bf16 input and output cast boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _identity_mask_softmax_kernel(
    input_ptr,
    output_ptr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N_COLS + cols[None, :]
    mask = cols[None, :] < N_COLS

    scores = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(scores, axis=1)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    out = numer / denom[:, None]

    tl.store(output_ptr + offsets, out.to(tl.bfloat16), mask=mask)


# (T([192,512,512], bf16), S([8,1,512,512]), S([-1,24,512,512]), S([-1,512,512]))
@oracle_impl(hardware="B200", point="4f884edd", BLOCK_M=16, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    x, _mask_shape, _view_shape, _out_shape = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_rows = x.numel() // int(x.shape[-1])
    n_cols = int(x.shape[-1])

    _identity_mask_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        x,
        out,
        N_COLS=n_cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
