"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvBert training pointwise/layout scope by returning the split-head permute as the required metadata-only view over the input projection and materializing the dense `[16384,384]` clone output directly from Inductor's broadcast add, explicit fp32-to-bf16 cast, and multiply expression, whereas Inductor lowers the broadcast add/cast, permute, multiply, clone, view, and split-head permute as generic pointwise and layout work; Inductor cannot do this today because it does not recognize this fixed ConvBert QKV product plus aliasing split-head output as a single guarded layout/product template; the fix is SCHEDULER_FUSION: add a guarded ConvBert layout-product lowering that preserves the aliasing output and writes the contiguous product in one tiled pass."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _cast_add_mul_kernel(
    activation_ptr,
    bias_ptr,
    projection_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = cols < 384
    batch = rows // 512
    seq = rows - batch * 512

    activation = tl.load(
        activation_ptr + batch * 196608 + cols * 512 + seq,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    projection = tl.load(
        projection_ptr + rows * 384 + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    rounded_add = (activation + bias).to(tl.bfloat16).to(tl.float32)
    out = rounded_add * projection
    tl.store(out_ptr + rows * 384 + cols, out, mask=mask)


@oracle_impl(hardware="B200", point="06a51e7f", BLOCK_M=16, BLOCK_C=128, num_warps=1)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_C, num_warps):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    batch = int(_shape_param_0[0])
    seq = int(_shape_param_0[1])
    channels = int(_shape_param_0[2])
    n = int(_shape_param_2[0])
    width = int(_shape_param_2[1])
    head_view = arg2_1.as_strided(
        (batch, channels // 64, seq, 64),
        (seq * channels, 64, channels, 1),
    )
    out = torch.empty_like(arg2_1)

    _cast_add_mul_kernel[(triton.cdiv(width, BLOCK_C), triton.cdiv(n, BLOCK_M))](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, head_view
