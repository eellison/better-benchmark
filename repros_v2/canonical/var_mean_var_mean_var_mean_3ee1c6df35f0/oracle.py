"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RepVGG bf16 triple training-BatchNorm branch-sum scope, including three fp32 population var_mean reductions over channels-last inputs, eps=1e-5 rsqrt side outputs, six mutable running-stat copy_ aliases with the captured variance-correction literal, exact fp32 affine expressions followed by per-branch bf16 casts, the captured bf16 add order, NaN-preserving bf16 ReLU, and the returned saved-mean views, whereas Inductor lowers the sibling BN-training reductions, mutable stat updates, and shared branch-sum activation through separate generic normalization and pointwise schedules; Inductor cannot do this today because the normalization scheduler does not co-schedule three mutable BN-training templates while sinking their exact bf16 epilogues into one full-scope activation store; the fix is SCHEDULER_FUSION: extend the BN-training scheduler to group sibling channel-stat reductions, emit running-stat side effects, and fuse the exact bf16 branch-sum ReLU epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _triple_partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    partial_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    e_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = (e_offsets[:, None] < E) & (c_offsets[None, :] < C)
    flat = e_offsets[:, None] * C + c_offsets[None, :]

    x0 = tl.load(x0_ptr + flat, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + flat, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + flat, mask=mask, other=0.0).to(tl.float32)
    active0 = tl.where(mask, x0, 0.0)
    active1 = tl.where(mask, x1, 0.0)
    active2 = tl.where(mask, x2, 0.0)

    weights = tl.where(mask, 1.0, 0.0)
    zero = tl.zeros((BLOCK_E, BLOCK_C), tl.float32)
    mean0, m20, _ = triton_helpers.welford(active0, zero, weights, 0)
    mean1, m21, _ = triton_helpers.welford(active1, zero, weights, 0)
    mean2, m22, _ = triton_helpers.welford(active2, zero, weights, 0)

    chunk = tl.program_id(1)
    offsets = chunk * C + c_offsets
    plane = NUM_CHUNKS * C
    c_mask = c_offsets < C
    tl.store(partial_ptr + offsets, mean0, mask=c_mask)
    tl.store(partial_ptr + plane + offsets, mean1, mask=c_mask)
    tl.store(partial_ptr + 2 * plane + offsets, mean2, mask=c_mask)
    tl.store(partial_ptr + 3 * plane + offsets, m20, mask=c_mask)
    tl.store(partial_ptr + 4 * plane + offsets, m21, mask=c_mask)
    tl.store(partial_ptr + 5 * plane + offsets, m22, mask=c_mask)


@triton.jit
def _triple_finalize_stats_kernel(
    partial_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    running_mean2_ptr,
    running_var2_ptr,
    mean0_ptr,
    mean1_ptr,
    mean2_ptr,
    invstd0_ptr,
    invstd1_ptr,
    invstd2_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (c_offsets[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + c_offsets[:, None]
    plane = NUM_CHUNKS * C

    chunk_mean0 = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_mean1 = tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_mean2 = tl.load(partial_ptr + 2 * plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_m20 = tl.load(partial_ptr + 3 * plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_m21 = tl.load(partial_ptr + 4 * plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_m22 = tl.load(partial_ptr + 5 * plane + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_start = chunks * BLOCK_E
    counts = tl.where(chunks < NUM_CHUNKS, tl.minimum(BLOCK_E, E - chunk_start), 0).to(tl.float32)
    weights = tl.broadcast_to(counts[None, :], (BLOCK_C, BLOCK_CHUNKS))

    mean0, m20, _ = triton_helpers.welford(chunk_mean0, chunk_m20, weights, 1)
    mean1, m21, _ = triton_helpers.welford(chunk_mean1, chunk_m21, weights, 1)
    mean2, m22, _ = triton_helpers.welford(chunk_mean2, chunk_m22, weights, 1)
    var0 = _f32_mul(m20, 1.0 / E)
    var1 = _f32_mul(m21, 1.0 / E)
    var2 = _f32_mul(m22, 1.0 / E)
    invstd0 = 1.0 / libdevice.sqrt(_f32_add(var0, 1.0e-5))
    invstd1 = 1.0 / libdevice.sqrt(_f32_add(var1, 1.0e-5))
    invstd2 = 1.0 / libdevice.sqrt(_f32_add(var2, 1.0e-5))

    c_mask = c_offsets < C
    old_mean0 = tl.load(running_mean0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_mean2 = tl.load(running_mean2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var2 = tl.load(running_var2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    mean_update0 = _f32_mul(mean0, 0.1)
    mean_update1 = _f32_mul(mean1, 0.1)
    mean_update2 = _f32_mul(mean2, 0.1)
    var_update0 = _f32_mul(_f32_mul(var0, 1.0000398612827361), 0.1)
    var_update1 = _f32_mul(_f32_mul(var1, 1.0000398612827361), 0.1)
    var_update2 = _f32_mul(_f32_mul(var2, 1.0000398612827361), 0.1)

    tl.store(running_mean0_ptr + c_offsets, _f32_add(mean_update0, _f32_mul(old_mean0, 0.9)), mask=c_mask)
    tl.store(running_var0_ptr + c_offsets, _f32_add(var_update0, _f32_mul(old_var0, 0.9)), mask=c_mask)
    tl.store(running_mean1_ptr + c_offsets, _f32_add(mean_update1, _f32_mul(old_mean1, 0.9)), mask=c_mask)
    tl.store(running_var1_ptr + c_offsets, _f32_add(var_update1, _f32_mul(old_var1, 0.9)), mask=c_mask)
    tl.store(running_mean2_ptr + c_offsets, _f32_add(mean_update2, _f32_mul(old_mean2, 0.9)), mask=c_mask)
    tl.store(running_var2_ptr + c_offsets, _f32_add(var_update2, _f32_mul(old_var2, 0.9)), mask=c_mask)

    tl.store(mean0_ptr + c_offsets, mean0, mask=c_mask)
    tl.store(mean1_ptr + c_offsets, mean1, mask=c_mask)
    tl.store(mean2_ptr + c_offsets, mean2, mask=c_mask)
    tl.store(invstd0_ptr + c_offsets, invstd0, mask=c_mask)
    tl.store(invstd1_ptr + c_offsets, invstd1, mask=c_mask)
    tl.store(invstd2_ptr + c_offsets, invstd2, mask=c_mask)


@triton.jit
def _triple_bn_sum_relu_kernel(
    x0_ptr,
    weight0_ptr,
    bias0_ptr,
    x1_ptr,
    weight1_ptr,
    bias1_ptr,
    x2_ptr,
    weight2_ptr,
    bias2_ptr,
    mean0_ptr,
    mean1_ptr,
    mean2_ptr,
    invstd0_ptr,
    invstd1_ptr,
    invstd2_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channels = offsets % C

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd0 = tl.load(invstd0_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd1 = tl.load(invstd1_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd2 = tl.load(invstd2_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + channels, mask=mask, other=0.0).to(tl.float32)

    y0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x0, mean0), invstd0), weight0), bias0).to(tl.bfloat16)
    y1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1), invstd1), weight1), bias1).to(tl.bfloat16)
    y2 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x2, mean2), invstd2), weight2), bias2).to(tl.bfloat16)
    add12 = _f32_add(y1.to(tl.float32), y2.to(tl.float32)).to(tl.bfloat16)
    added = _f32_add(add12.to(tl.float32), y0.to(tl.float32)).to(tl.bfloat16)
    zero = tl.full((BLOCK,), 0.0, tl.bfloat16)
    relu = tl.where(added < zero, zero, added)
    tl.store(out_ptr + offsets, relu, mask=mask)


@oracle_impl(hardware="B200", point="3009c407", block_e=1024, block_c=8, block=1024)
@oracle_impl(hardware="B200", point="9a6632a5", block_e=1024, block_c=8, block=1024)
def oracle_forward(inputs, *, block_e: int, block_c: int, block: int):
    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        x2,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = inputs

    n, c, h, w = x0.shape
    e = n * h * w
    total = e * c
    num_chunks = triton.cdiv(e, block_e)
    block_chunks = _next_power_of_2(num_chunks)

    mean0 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean1 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean2 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd0 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    invstd1 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    invstd2 = torch.empty((c,), device=x0.device, dtype=torch.float32)
    partial = torch.empty((6, num_chunks, c), device=x0.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=torch.bfloat16)

    _triple_partial_stats_kernel[(triton.cdiv(c, block_c), num_chunks)](
        x0,
        x1,
        x2,
        partial,
        c,
        e,
        num_chunks,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        num_warps=8,
        num_stages=3,
    )
    _triple_finalize_stats_kernel[(triton.cdiv(c, block_c),)](
        partial,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
        mean0,
        mean1,
        mean2,
        invstd0,
        invstd1,
        invstd2,
        c,
        e,
        num_chunks,
        BLOCK_E=block_e,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=block_c,
        num_warps=1,
        num_stages=3,
    )
    _triple_bn_sum_relu_kernel[(triton.cdiv(total, block),)](
        x0,
        weight0,
        bias0,
        x1,
        weight1,
        bias1,
        x2,
        weight2,
        bias2,
        mean0,
        mean1,
        mean2,
        invstd0,
        invstd1,
        invstd2,
        out,
        total,
        c,
        BLOCK=block,
        num_warps=4,
        num_stages=3,
    )

    return (
        invstd0,
        invstd1,
        invstd2,
        out,
        mean2,
        mean1,
        mean0,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
    )


@oracle_impl(hardware="B200", point="d6dd53ac", block=1024)
def oracle_forward_d6dd53ac(inputs, *, block: int):
    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        x2,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = inputs

    c = x0.shape[1]
    total = x0.numel()
    f0 = torch.ops.prims.convert_element_type.default(x0, torch.float32)
    f1 = torch.ops.prims.convert_element_type.default(x1, torch.float32)
    f2 = torch.ops.prims.convert_element_type.default(x2, torch.float32)
    var0, mean0 = torch.ops.aten.var_mean.correction(f0, [0, 2, 3], correction=0, keepdim=True)
    var1, mean1 = torch.ops.aten.var_mean.correction(f1, [0, 2, 3], correction=0, keepdim=True)
    var2, mean2 = torch.ops.aten.var_mean.correction(f2, [0, 2, 3], correction=0, keepdim=True)
    invstd0 = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var0, 1.0e-5))
    invstd1 = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var1, 1.0e-5))
    invstd2 = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var2, 1.0e-5))
    mean0_1d = torch.ops.aten.squeeze.dims(mean0, [0, 2, 3])
    mean1_1d = torch.ops.aten.squeeze.dims(mean1, [0, 2, 3])
    mean2_1d = torch.ops.aten.squeeze.dims(mean2, [0, 2, 3])
    var0_1d = torch.ops.aten.squeeze.dims(var0, [0, 2, 3])
    var1_1d = torch.ops.aten.squeeze.dims(var1, [0, 2, 3])
    var2_1d = torch.ops.aten.squeeze.dims(var2, [0, 2, 3])
    invstd0_1d = torch.ops.aten.squeeze.dims(invstd0, [0, 2, 3])
    invstd1_1d = torch.ops.aten.squeeze.dims(invstd1, [0, 2, 3])
    invstd2_1d = torch.ops.aten.squeeze.dims(invstd2, [0, 2, 3])

    running_mean0.copy_(mean0_1d * 0.1 + running_mean0 * 0.9)
    running_var0.copy_(var0_1d * 1.0000398612827361 * 0.1 + running_var0 * 0.9)
    running_mean1.copy_(mean1_1d * 0.1 + running_mean1 * 0.9)
    running_var1.copy_(var1_1d * 1.0000398612827361 * 0.1 + running_var1 * 0.9)
    running_mean2.copy_(mean2_1d * 0.1 + running_mean2 * 0.9)
    running_var2.copy_(var2_1d * 1.0000398612827361 * 0.1 + running_var2 * 0.9)

    out = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=torch.bfloat16)
    _triple_bn_sum_relu_kernel[(triton.cdiv(total, block),)](
        x0,
        weight0,
        bias0,
        x1,
        weight1,
        bias1,
        x2,
        weight2,
        bias2,
        mean0,
        mean1,
        mean2,
        invstd0_1d,
        invstd1_1d,
        invstd2_1d,
        out,
        total,
        c,
        BLOCK=block,
        num_warps=4,
        num_stages=3,
    )
    return (
        invstd0_1d,
        invstd1_1d,
        invstd2_1d,
        out,
        mean2,
        mean1,
        mean0,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
    )
