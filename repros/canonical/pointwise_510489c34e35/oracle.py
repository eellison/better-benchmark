"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Visformer Q/K/V layout split by reading the strided channels-last source view once and materializing the three required contiguous clone outputs in one Triton multi-output layout kernel, whereas Inductor lowers the view/permute/unbind/clone chain as sibling generic layout-copy materializations; Inductor cannot do this today because the scheduler does not fuse multiple clone users of the same unbound strided view into one producer with separate output stores; the fix is SCHEDULER_FUSION: teach layout-copy scheduling to group sibling unbind clone materializations and emit a fused multi-output affine copy while preserving the fresh contiguous output storages."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _qkv_clone_layout_kernel(
    input_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    p_block = tl.program_id(1)
    d_block = tl.program_id(2)

    p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)[:, None]
    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    mask = (row < ROWS) & (p < P) & (d < D)

    batch = row // HEADS
    head = row - batch * HEADS
    qkv_stride = HEADS * D * CHANNEL_STRIDE
    input_base = batch * BATCH_STRIDE + p * POS_STRIDE + (head * D + d) * CHANNEL_STRIDE
    output_offsets = row * P * D + p * D + d

    q_values = tl.load(input_ptr + input_base, mask=mask, other=0.0)
    k_values = tl.load(input_ptr + input_base + qkv_stride, mask=mask, other=0.0)
    v_values = tl.load(input_ptr + input_base + 2 * qkv_stride, mask=mask, other=0.0)

    tl.store(q_out_ptr + output_offsets, q_values, mask=mask)
    tl.store(k_out_ptr + output_offsets, k_values, mask=mask)
    tl.store(v_out_ptr + output_offsets, v_values, mask=mask)


# 9cb825ed: (T([128,2304,7,7], bf16, stride=(112896,1,16128,2304)), S([128,3,6,128,49]))
@oracle_impl(hardware="B200", point="9cb825ed", BLOCK_P=32, BLOCK_D=64, num_warps=8)
# 97e22389: (T([128,1152,14,14], bf16, stride=(225792,1,16128,1152)), S([128,3,6,64,196]))
@oracle_impl(hardware="B200", point="97e22389", BLOCK_P=32, BLOCK_D=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_P: int, BLOCK_D: int, num_warps: int):
    arg0_1, shape0 = inputs
    batch, _qkv, heads, d_model, positions = (int(dim) for dim in shape0)
    out_shape = (batch, heads, positions, d_model)

    q_out = torch.empty(out_shape, device=arg0_1.device, dtype=arg0_1.dtype)
    k_out = torch.empty(out_shape, device=arg0_1.device, dtype=arg0_1.dtype)
    v_out = torch.empty(out_shape, device=arg0_1.device, dtype=arg0_1.dtype)

    rows = batch * heads
    grid = (
        rows,
        triton.cdiv(positions, BLOCK_P),
        triton.cdiv(d_model, BLOCK_D),
    )
    _qkv_clone_layout_kernel[grid](
        arg0_1,
        q_out,
        k_out,
        v_out,
        ROWS=rows,
        HEADS=heads,
        D=d_model,
        P=positions,
        BATCH_STRIDE=arg0_1.stride(0),
        CHANNEL_STRIDE=arg0_1.stride(1),
        POS_STRIDE=arg0_1.stride(3),
        BLOCK_P=BLOCK_P,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=3,
    )
    return q_out, k_out, v_out
