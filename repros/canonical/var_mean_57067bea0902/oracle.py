"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16-input add-plus-affine LayerNorm scope in one shape-specialized Triton row kernel, including the `[4096,4096] -> [8,512,4096]` view, bf16-to-fp32 residual add, correction=0 var_mean over hidden size 4096, `rsqrt(var + 1e-12)`, normalized fp32 side output, affine fp32 side output, final bf16 `[4096,4096]` view, and sibling `rsqrt / 4096` output, whereas Inductor currently emits a generic fused var_mean/LayerNorm schedule for the same decomposed scope; Inductor cannot do this today because its normalization scheduler does not expose a reusable full-scope row template that keeps the row tile resident through all live normalized, affine, cast, and saved-scale epilogues; the fix is SCHEDULER_FUSION: add a guarded LayerNorm row schedule that preserves correction=0 Welford numerics while fusing the complete multi-output epilogue contract."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _add_layernorm_full_kernel(
    flat_bf16_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    normalized_ptr,
    affine_ptr,
    final_bf16_ptr,
    invstd_div_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_bf16_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    x = flat + residual

    mean_acc = tl.zeros([BLOCK_H], tl.float32)
    m2_acc = tl.zeros([BLOCK_H], tl.float32)
    weight_acc = tl.zeros([BLOCK_H], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(mask, mean_next, mean_acc)
    m2_acc = tl.where(mask, m2_next, m2_acc)
    weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)
    variance = m2 / HIDDEN

    centered = x - mean
    invstd = libdevice.rsqrt(variance + 1.0e-12)
    normalized = centered * invstd

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    affine = normalized * weight + bias

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(affine_ptr + offsets, affine, mask=mask)
    tl.store(final_bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(invstd_div_ptr + row, invstd / 4096.0)


# ba44cc6a: (T([4096,4096], bf16), T([8,512,4096], f32), T([4096], f32), T([4096], f32), ...)
@oracle_impl(hardware="B200", point="ba44cc6a", BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs

    normalized = torch.empty_strided(
        (8, 512, 4096),
        (2097152, 4096, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    affine = torch.empty_strided(
        (8, 512, 4096),
        (2097152, 4096, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        (4096, 4096),
        (4096, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        (8, 512, 1),
        (512, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _add_layernorm_full_kernel[(4096,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        normalized,
        affine,
        final_bf16,
        invstd_div,
        HIDDEN=4096,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return normalized, affine, final_bf16, invstd_div
