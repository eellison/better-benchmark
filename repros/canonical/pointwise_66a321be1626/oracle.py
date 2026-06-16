"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete VisFormer bf16 Q/K/V layout split, including the padded Q and K tensors, the contiguous V materialization, and the final eager-compatible permuted views, in two Triton multi-output layout kernels, whereas Inductor lowers the sibling unbind/permute/clone/view/constant_pad_nd materializations as separate generic layout-copy and padding work; Inductor cannot do this today because its scheduler does not fuse multiple users of the same strided QKV producer when each output has a different affine store map and padding domain; the fix is SCHEDULER_FUSION: teach layout-copy scheduling to emit a multi-output fused producer for sibling Q/K/V materializations with per-output affine store maps and zero-fill regions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _qv_padded_layout_kernel(
    input_ptr,
    qpad_ptr,
    qbase_ptr,
    vbase_ptr,
    XBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    d = xindex % 128
    p = (xindex // 128) % 56
    row = xindex // 7168
    value_mask = p < 49

    batch = row // 6
    head = row - batch * 6
    input_base = batch * 112896 + p * 2304 + head * 128 + d

    q_values = tl.load(input_ptr + input_base, mask=value_mask, other=0.0)
    v_values = tl.load(input_ptr + input_base + 1536, mask=value_mask, other=0.0)

    base_offsets = row * 6272 + p * 128 + d

    tl.store(qpad_ptr + xindex, q_values)
    tl.store(qbase_ptr + base_offsets, q_values, mask=value_mask)
    tl.store(vbase_ptr + base_offsets, v_values, mask=value_mask)


@triton.jit
def _k_padded_layout_kernel(
    input_ptr,
    kpad_ptr,
    kbase_ptr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    yindex = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    mask = xindex < 56
    value_mask = xindex < 49

    d = yindex % 128
    row = yindex // 128
    batch = row // 6
    head = row - batch * 6
    input_offsets = 768 + d + head * 128 + xindex * 2304 + batch * 112896
    k_values = tl.load(input_ptr + input_offsets, mask=value_mask, other=0.0)

    tl.store(kpad_ptr + yindex * 56 + xindex, k_values, mask=mask)
    tl.store(kbase_ptr + yindex * 49 + xindex, k_values, mask=value_mask)


@oracle_impl(hardware="B200", point="9cb825ed", QV_XBLOCK=1024, K_YBLOCK=16, K_XBLOCK=64)
def oracle_forward(inputs, *, QV_XBLOCK, K_YBLOCK, K_XBLOCK):
    source = inputs[0]
    qpad = torch.empty((768, 56, 128), device=source.device, dtype=source.dtype)
    kpad = torch.empty((768, 128, 56), device=source.device, dtype=source.dtype)
    qbase = torch.empty((768, 49, 128), device=source.device, dtype=source.dtype)
    kbase = torch.empty((768, 128, 49), device=source.device, dtype=source.dtype)
    vbase = torch.empty((768, 49, 128), device=source.device, dtype=source.dtype)

    _qv_padded_layout_kernel[(triton.cdiv(768 * 56 * 128, QV_XBLOCK),)](
        source,
        qpad,
        qbase,
        vbase,
        XBLOCK=QV_XBLOCK,
        num_warps=4,
    )
    _k_padded_layout_kernel[(triton.cdiv(56, K_XBLOCK), triton.cdiv(768 * 128, K_YBLOCK))](
        source,
        kpad,
        kbase,
        YBLOCK=K_YBLOCK,
        XBLOCK=K_XBLOCK,
        num_warps=4,
    )
    return (
        qpad,
        kpad,
        vbase,
        vbase.permute(0, 2, 1),
        qbase.permute(0, 2, 1),
        kbase.permute(0, 2, 1),
    )
