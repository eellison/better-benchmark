"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BERT bf16 `[16,2]` log-softmax-and-cast region in one tiny-K Triton kernel, preserving the repro's bf16-to-fp32 input cast, row amax, subtract, natural exp, sum, natural log, final subtract, and visible bf16 output rounding, whereas Inductor lowers the same amax/sub/exp/sum/log/sub/cast idiom through its generic small-reduction and pointwise schedule; Inductor cannot do this today because its pattern library does not recognize a two-column log-softmax with a final bf16 boundary as a shape-guarded straight-line reduction template; the fix is NEW_PATTERN: add a guarded K=2 log-softmax lowering that emits the stable fp32 row math and bf16 output stores in one compact kernel when benchmarking shows it beats or ties the generic schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _log_softmax_k2_bf16_kernel(
    x_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    OUT_STRIDE_0: tl.constexpr,
    OUT_STRIDE_1: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < ROWS

    x0 = tl.load(x_ptr + rows * X_STRIDE_0, mask=mask, other=-float("inf")).to(tl.float32)
    x1 = tl.load(x_ptr + rows * X_STRIDE_0 + X_STRIDE_1, mask=mask, other=-float("inf")).to(tl.float32)

    row_max = tl.maximum(x0, x1)
    shifted0 = x0 - row_max
    shifted1 = x1 - row_max
    denom = libdevice.exp(shifted0) + libdevice.exp(shifted1)
    log_denom = libdevice.log(denom)

    tl.store(
        out_ptr + rows * OUT_STRIDE_0,
        (shifted0 - log_denom).to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        out_ptr + rows * OUT_STRIDE_0 + OUT_STRIDE_1,
        (shifted1 - log_denom).to(tl.bfloat16),
        mask=mask,
    )


# de033194: BERT bf16 log_softmax over 16 rows of width 2 with bf16 return.
@oracle_impl(hardware="B200", point="de033194", block_m=16, num_warps=1, num_stages=3)
def oracle_forward(inputs, *, block_m: int, num_warps: int, num_stages: int):
    (arg0_1,) = inputs
    rows = int(arg0_1.shape[0])
    out = torch.empty_strided(
        (rows, 2),
        (2, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _log_softmax_k2_bf16_kernel[(triton.cdiv(rows, block_m),)](
        arg0_1,
        out,
        ROWS=rows,
        X_STRIDE_0=arg0_1.stride(0),
        X_STRIDE_1=arg0_1.stride(1),
        OUT_STRIDE_0=out.stride(0),
        OUT_STRIDE_1=out.stride(1),
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
