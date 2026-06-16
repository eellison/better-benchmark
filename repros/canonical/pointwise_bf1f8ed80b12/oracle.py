"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Visformer Q/K/V layout split, including the three fresh padded Q/K/V tensors and the three eager-compatible permuted view outputs from separate fresh clone storages, by reading the strided channels-last source once per Q/K/V plane and materializing the required affine stores with Triton multi-output kernels, whereas Inductor lowers the sibling unbind/permute/clone/view/constant_pad_nd branches as separate generic layout-copy and padding kernels; Inductor cannot do this today because its scheduler does not fuse multiple users of the same unbound strided QKV producer when each returned output has a different store map, padding domain, and view-alias contract; the fix is SCHEDULER_FUSION: teach layout-copy scheduling to group Q/K/V sibling materializations and sink fixed zero-padding into per-output affine store maps while preserving fresh backing storages and returned view strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _qv_padded_layout_kernel(
    input_ptr,
    qpad_ptr,
    vpad_ptr,
    qbase_ptr,
    vbase_ptr,
    total_pad_elements: tl.constexpr,
    heads: tl.constexpr,
    d_model: tl.constexpr,
    positions: tl.constexpr,
    padded_positions: tl.constexpr,
    batch_stride: tl.constexpr,
    channel_stride: tl.constexpr,
    position_stride: tl.constexpr,
    qkv_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_pad_elements

    d = offsets % d_model
    p = (offsets // d_model) % padded_positions
    row = offsets // (padded_positions * d_model)
    value_mask = mask & (p < positions)

    batch = row // heads
    head = row - batch * heads
    input_base = (
        batch * batch_stride
        + p * position_stride
        + (head * d_model + d) * channel_stride
    )

    q_values = tl.load(input_ptr + input_base, mask=value_mask, other=0.0)
    v_values = tl.load(input_ptr + input_base + 2 * qkv_stride, mask=value_mask, other=0.0)
    base_offsets = row * positions * d_model + p * d_model + d

    tl.store(qpad_ptr + offsets, q_values, mask=mask)
    tl.store(vpad_ptr + offsets, v_values, mask=mask)
    tl.store(qbase_ptr + base_offsets, q_values, mask=value_mask)
    tl.store(vbase_ptr + base_offsets, v_values, mask=value_mask)


@triton.jit
def _k_padded_layout_kernel(
    input_ptr,
    kpad_ptr,
    kbase_ptr,
    heads: tl.constexpr,
    d_model: tl.constexpr,
    positions: tl.constexpr,
    padded_positions: tl.constexpr,
    batch_stride: tl.constexpr,
    channel_stride: tl.constexpr,
    position_stride: tl.constexpr,
    qkv_stride: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    p = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    y_mask = y < (128 * heads * d_model)
    p_mask = p < padded_positions
    mask = y_mask & p_mask
    value_mask = mask & (p < positions)

    d = y % d_model
    row = y // d_model
    batch = row // heads
    head = row - batch * heads
    input_offsets = (
        batch * batch_stride
        + p * position_stride
        + qkv_stride
        + (head * d_model + d) * channel_stride
    )
    k_values = tl.load(input_ptr + input_offsets, mask=value_mask, other=0.0)

    tl.store(kpad_ptr + y * padded_positions + p, k_values, mask=mask)
    tl.store(kbase_ptr + y * positions + p, k_values, mask=value_mask)


# 97e22389: (T([128,1152,14,14], bf16, stride=(225792,1,16128,1152)), S([128,3,6,64,196]), ...)
@oracle_impl(hardware="B200", point="97e22389", QV_BLOCK=1024, K_YBLOCK=16, K_XBLOCK=64)
def oracle_forward(inputs, *, QV_BLOCK: int, K_YBLOCK: int, K_XBLOCK: int):
    source = inputs[0]
    batch, _qkv, heads, d_model, positions = (int(dim) for dim in inputs[1])
    padded_positions = positions + int(inputs[6][3])
    rows = batch * heads

    qpad = torch.empty((rows, padded_positions, d_model), device=source.device, dtype=source.dtype)
    kpad = torch.empty((rows, d_model, padded_positions), device=source.device, dtype=source.dtype)
    vpad = torch.empty((rows, padded_positions, d_model), device=source.device, dtype=source.dtype)
    qbase = torch.empty((rows, positions, d_model), device=source.device, dtype=source.dtype)
    kbase = torch.empty((rows, d_model, positions), device=source.device, dtype=source.dtype)
    vbase = torch.empty((rows, positions, d_model), device=source.device, dtype=source.dtype)

    qkv_stride = heads * d_model * source.stride(1)
    total_qv_pad = rows * padded_positions * d_model
    _qv_padded_layout_kernel[(triton.cdiv(total_qv_pad, QV_BLOCK),)](
        source,
        qpad,
        vpad,
        qbase,
        vbase,
        total_pad_elements=total_qv_pad,
        heads=heads,
        d_model=d_model,
        positions=positions,
        padded_positions=padded_positions,
        batch_stride=source.stride(0),
        channel_stride=source.stride(1),
        position_stride=source.stride(3),
        qkv_stride=qkv_stride,
        BLOCK=QV_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _k_padded_layout_kernel[
        (triton.cdiv(padded_positions, K_XBLOCK), triton.cdiv(rows * d_model, K_YBLOCK))
    ](
        source,
        kpad,
        kbase,
        heads=heads,
        d_model=d_model,
        positions=positions,
        padded_positions=padded_positions,
        batch_stride=source.stride(0),
        channel_stride=source.stride(1),
        position_stride=source.stride(3),
        qkv_stride=qkv_stride,
        YBLOCK=K_YBLOCK,
        XBLOCK=K_XBLOCK,
        num_warps=4,
        num_stages=3,
    )
    return (
        qpad,
        kpad,
        vpad,
        vbase.permute(0, 2, 1),
        qbase.permute(0, 2, 1),
        kbase.permute(0, 2, 1),
    )
