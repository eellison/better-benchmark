"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MobileViT QKV reshape/permute/unbind plus three right-pad outputs by writing Q, K, and V padded layouts directly from the packed projection in one Triton kernel, whereas Inductor lowers the sibling constant_pad_nd materializations as generic branch-local output work after the view split; Inductor cannot do this today because its scheduler does not group multiple layout/pad materializations from one unbound QKV producer into a single multi-output copy plan; the fix is SCHEDULER_FUSION: teach pointwise/layout-copy scheduling to fuse sibling QKV split consumers and sink fixed zero-pad epilogues into one generated multi-output kernel."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _qkv_right_pad_kernel(
    in_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    N_ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUT_DIM: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, COL_BLOCK)
    row_mask = rows < N_ROWS
    col_mask = cols < OUT_DIM

    token = rows % SEQ
    tmp = rows // SEQ
    head = tmp % HEADS
    batch = tmp // HEADS

    valid = row_mask[:, None] & (cols[None, :] < HEAD_DIM)
    qkv_plane: tl.constexpr = HEADS * HEAD_DIM
    input_row = batch * SEQ + token
    input_base = (
        input_row[:, None] * (3 * qkv_plane)
        + head[:, None] * HEAD_DIM
        + cols[None, :]
    )

    q_values = tl.load(in_ptr + input_base, mask=valid, other=0.0)
    k_values = tl.load(in_ptr + input_base + qkv_plane, mask=valid, other=0.0)
    v_values = tl.load(in_ptr + input_base + 2 * qkv_plane, mask=valid, other=0.0)

    output_offsets = rows[:, None] * OUT_DIM + cols[None, :]
    store_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(q_out_ptr + output_offsets, q_values, mask=store_mask)
    tl.store(k_out_ptr + output_offsets, k_values, mask=store_mask)
    tl.store(v_out_ptr + output_offsets, v_values, mask=store_mask)


@oracle_impl(
    hardware="B200",
    point="a07ff8de",
    SEQ=16,
    HEADS=4,
    HEAD_DIM=60,
    OUT_DIM=64,
    ROW_BLOCK=16,
    COL_BLOCK=64,
    num_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="419b45cb",
    SEQ=256,
    HEADS=4,
    HEAD_DIM=36,
    OUT_DIM=40,
    ROW_BLOCK=16,
    COL_BLOCK=64,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    SEQ: int,
    HEADS: int,
    HEAD_DIM: int,
    OUT_DIM: int,
    ROW_BLOCK: int,
    COL_BLOCK: int,
    num_warps: int,
):
    arg0_1, shape0, _shape1, _pad0, _pad1, _pad2 = inputs
    batch = int(shape0[0])
    out_shape = (batch, HEADS, SEQ, OUT_DIM)
    out_stride = (HEADS * SEQ * OUT_DIM, SEQ * OUT_DIM, OUT_DIM, 1)

    q_out = torch.empty_strided(
        out_shape,
        out_stride,
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    k_out = torch.empty_strided(
        out_shape,
        out_stride,
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    v_out = torch.empty_strided(
        out_shape,
        out_stride,
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    n_rows = batch * HEADS * SEQ
    _qkv_right_pad_kernel[(triton.cdiv(n_rows, ROW_BLOCK),)](
        arg0_1,
        q_out,
        k_out,
        v_out,
        N_ROWS=n_rows,
        SEQ=SEQ,
        HEADS=HEADS,
        HEAD_DIM=HEAD_DIM,
        OUT_DIM=OUT_DIM,
        ROW_BLOCK=ROW_BLOCK,
        COL_BLOCK=COL_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return q_out, k_out, v_out
