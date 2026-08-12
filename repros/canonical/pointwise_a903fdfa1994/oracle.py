"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MobileViT view/view/permute/full/slice_scatter scope by materializing the visible all-zero `[512,4,16,64]` base output and the separate padded slice_scatter output in one storage-linear Triton pass, whereas Inductor lowers the reshaped permute producer, fresh zero full, and functional slice update through generic pointwise/layout scheduling; Inductor cannot do this today because its scheduler does not sink fixed view/permute algebra through a zero-base slice_scatter into multi-output direct stores while preserving the returned pre-scatter base tensor; the fix is SCHEDULER_FUSION: teach layout scheduling to recognize static MobileViT padded-head slice_scatter patterns and emit direct multi-output materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 512
HEADS = 4
SEQ = 16
ACTIVE_DIM = 60
PADDED_DIM = 64
HIDDEN = HEADS * ACTIVE_DIM
ROWS = BATCH * HEADS * SEQ


@triton.jit
def _mobilevit_zero_base_slice_scatter_kernel(
    src_ptr,
    base_ptr,
    out_ptr,
    rows_total: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    active_dim: tl.constexpr,
    padded_dim: tl.constexpr,
    hidden: tl.constexpr,
    block_rows: tl.constexpr,
    block_dim: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    dim = tl.arange(0, block_dim)
    row_mask = rows < rows_total
    dim_mask = dim < padded_dim
    mask = row_mask[:, None] & dim_mask[None, :]

    batch = rows // (heads * seq_len)
    rem = rows - batch * (heads * seq_len)
    head = rem // seq_len
    seq = rem - head * seq_len

    from_src = dim < active_dim
    src_offsets = (
        (batch[:, None] * seq_len + seq[:, None]) * hidden
        + head[:, None] * active_dim
        + dim[None, :]
    )
    out_offsets = rows[:, None] * padded_dim + dim[None, :]

    zeros = tl.full((block_rows, block_dim), 0.0, tl.float32)
    values = tl.load(src_ptr + src_offsets, mask=mask & from_src[None, :], other=0.0)
    values = tl.where(from_src[None, :], values, zeros)

    tl.store(base_ptr + out_offsets, zeros, mask=mask)
    tl.store(out_ptr + out_offsets, values, mask=mask)


# 51fe7e77: MobileViT bf16 [8192,240] -> visible zero base and padded [512,4,16,64] slice_scatter.
@oracle_impl(hardware="B200", point="51fe7e77", BLOCK_ROWS=16, BLOCK_DIM=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_DIM: int, num_warps: int):
    arg0_1 = inputs[0]
    full = torch.empty_strided(
        (BATCH, HEADS, SEQ, PADDED_DIM),
        (HEADS * SEQ * PADDED_DIM, SEQ * PADDED_DIM, PADDED_DIM, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    slice_scatter = torch.empty_strided(
        (BATCH, HEADS, SEQ, PADDED_DIM),
        (HEADS * SEQ * PADDED_DIM, SEQ * PADDED_DIM, PADDED_DIM, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    _mobilevit_zero_base_slice_scatter_kernel[(triton.cdiv(ROWS, BLOCK_ROWS),)](
        arg0_1,
        full,
        slice_scatter,
        rows_total=ROWS,
        heads=HEADS,
        seq_len=SEQ,
        active_dim=ACTIVE_DIM,
        padded_dim=PADDED_DIM,
        hidden=HIDDEN,
        block_rows=BLOCK_ROWS,
        block_dim=BLOCK_DIM,
        num_warps=num_warps,
        num_stages=3,
    )
    return full, slice_scatter
