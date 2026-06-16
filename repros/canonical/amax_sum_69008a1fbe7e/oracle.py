"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete XGLM bf16 additive-bias attention softmax in one Triton row kernel, including the `[512,128,128] -> [32,16,128,128]` metadata view, broadcast `[32,1,128,128]` bf16 bias add, sentinel maximum, stable fp32 last-dimension amax/exp/sum/div, final bf16 cast, and dense output layout, whereas Inductor lowers the decomposed view/add/maximum/amax/sub/exp/sum/div/cast graph through generic reduction scheduling; Inductor cannot do this today because its pattern library does not recognize this K=128 additive-bias attention softmax as a reusable full-scope row template with the broadcast bias and layout-only views sunk into the reduction/store; the fix is NEW_PATTERN: add a guarded additive-bias attention-softmax lowering that fuses score construction, stable row reductions, and final bf16 materialization."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bias_softmax_kernel(
    scores_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    flat_bh = rows // 128
    query = rows - flat_bh * 128
    batch = flat_bh // 16

    scores_offsets = rows[:, None] * 128 + cols[None, :]
    bias_offsets = batch[:, None] * 16384 + query[:, None] * 128 + cols[None, :]

    x = tl.load(scores_ptr + scores_offsets).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets).to(tl.float32)
    scores = tl.maximum(x + bias, -3.3895313892515355e38)

    row_max = tl.max(scores, axis=1)
    numer = libdevice.exp(scores - row_max[:, None])
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]
    tl.store(out_ptr + scores_offsets, probs.to(tl.bfloat16))


@oracle_impl(
    hardware="B200",
    point="87c3ffc0",
    BLOCK_M=16,
    BLOCK_N=128,
    num_warps=4,
    num_stages=1,
)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps, num_stages):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        (16384, 128, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _bias_softmax_kernel[(triton.cdiv(arg0_1.shape[0] * arg0_1.shape[1], BLOCK_M),)](
        arg0_1,
        arg1_1,
        out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
