"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvBert pointwise/layout scope by returning the head-split permute as the required metadata-only view over the input projection and materializing the dense `[16384,384]` clone output directly from Inductor's fused `(arg1 + arg2) * arg0` expression with fp32 pointwise math and a bf16 store; Inductor lowers the permuted view, broadcast add, multiply, clone, and view as generic layout and pointwise work rather than recognizing the split-head alias plus dense epilogue as one small fixed-shape template; the fix is SCHEDULER_FUSION: add a guarded ConvBert QKV layout/product lowering that preserves the aliasing output and writes the contiguous product in one tiled pass."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _broadcast_add_mul_kernel(
    q_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = cols < 384
    batch = rows // 512
    seq = rows - batch * 512

    q = tl.load(q_ptr + rows * 384 + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + batch * 196608 + cols * 512 + seq, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

    out = (x + bias) * q
    tl.store(out_ptr + rows * 384 + cols, out, mask=mask)


@oracle_impl(hardware="B200", point="75c13bb9", BLOCK_M=16, BLOCK_C=64, num_warps=1)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_C, num_warps):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    n = int(_shape_param_2[0])
    c = int(_shape_param_2[1])
    batch = int(arg1_1.shape[0])
    seq = n // batch
    head_view = arg0_1.as_strided(
        (batch, c // 64, seq, 64),
        (seq * c, 64, c, 1),
    )
    out = torch.empty_like(arg0_1)

    _broadcast_add_mul_kernel[(triton.cdiv(c, BLOCK_C), triton.cdiv(n, BLOCK_M))](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return head_view, out
