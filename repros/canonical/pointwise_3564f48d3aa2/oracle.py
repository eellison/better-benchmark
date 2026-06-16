"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet inference concat -> BN affine -> bf16 cast -> ReLU scope in one Triton kernel, reading the six concat inputs through virtual channel selection and storing only the final contiguous bf16 output; Inductor materializes or schedules the fixed channel concat separately from the downstream per-channel affine and epilogue, and it cannot yet inline this multi-source concat producer into a channel-dependent pointwise consumer. The oracle preserves the captured numerics exactly, including fp32 sqrt(var + 1e-5), bf16 output rounding before ReLU, and NaN propagation from negative var + eps inputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    x4_ptr,
    x5_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C0: tl.constexpr,
    C_TOTAL: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    safe_offsets = tl.where(mask, offsets, 0)

    spatial = safe_offsets % HW
    channel = (safe_offsets // HW) % C_TOTAL
    batch = safe_offsets // (C_TOTAL * HW)
    rel = channel - C0

    in0 = channel < C0
    in1 = (rel >= 0) & (rel < 32)
    in2 = (rel >= 32) & (rel < 64)
    in3 = (rel >= 64) & (rel < 96)
    in4 = (rel >= 96) & (rel < 128)
    in5 = rel >= 128

    x0_offsets = (batch * C0 + channel) * HW + spatial
    branch_base = batch * 32 * HW + spatial
    x1_offsets = branch_base + rel * HW
    x2_offsets = branch_base + (rel - 32) * HW
    x3_offsets = branch_base + (rel - 64) * HW
    x4_offsets = branch_base + (rel - 96) * HW
    x5_offsets = branch_base + (rel - 128) * HW

    x = tl.load(x0_ptr + x0_offsets, mask=mask & in0, other=0.0).to(tl.float32)
    x += tl.load(x1_ptr + x1_offsets, mask=mask & in1, other=0.0).to(tl.float32)
    x += tl.load(x2_ptr + x2_offsets, mask=mask & in2, other=0.0).to(tl.float32)
    x += tl.load(x3_ptr + x3_offsets, mask=mask & in3, other=0.0).to(tl.float32)
    x += tl.load(x4_ptr + x4_offsets, mask=mask & in4, other=0.0).to(tl.float32)
    x += tl.load(x5_ptr + x5_offsets, mask=mask & in5, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * (1.0 / tl.sqrt(var + 1.0e-5))
    y = y * weight + bias
    y_bf16 = y.to(tl.bfloat16)
    zero = tl.full((BLOCK,), 0.0, tl.bfloat16)
    y_relu = tl.where(y_bf16 < zero, zero, y_bf16)
    tl.store(out_ptr + offsets, y_relu, mask=mask)


@oracle_impl(hardware="B200", point="a0c86fd4", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="64fc1db2", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="5ababadb", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="994d5473", BLOCK=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    x0, x1, x2, x3, x4, x5, mean, var, weight, bias = inputs
    n, c0, h, w = x0.shape
    c_total = mean.numel()
    hw = h * w
    out = torch.empty_strided(
        (n, c_total, h, w),
        (c_total * hw, hw, w, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    _cat_bn_relu_kernel[(triton.cdiv(out.numel(), BLOCK),)](
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        mean,
        var,
        weight,
        bias,
        out,
        TOTAL=out.numel(),
        C0=c0,
        C_TOTAL=c_total,
        HW=hw,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
