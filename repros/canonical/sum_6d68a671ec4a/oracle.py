"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle writes the returned bf16 QKV layout clone backing storage and produces the sibling bf16-rounded fp32 column sum from the same source-tensor tiles, whereas Inductor materializes the cat/view/permute/clone output and then rereads that buffer through separate reduction kernels; Inductor cannot do this today because its scheduler does not fuse a mandatory returned layout-copy producer with a compatible reduction consumer while preserving the aliasing transpose return and explicit bf16 round-trip on the sum; the fix is SCHEDULER_FUSION: teach layout materialization scheduling to emit same-tile reduction partials for returned clone buffers with sibling row-reduction users."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _materialize_partial_sum_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    matrix_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    CHANNELS: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_T: tl.constexpr,
    STRIDE_C: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)

    row_offsets = row_tile * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    col_offsets = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    source_cols = HEADS * CHANNELS
    source = col_offsets // source_cols
    source_inner = col_offsets - source * source_cols
    head = source_inner // CHANNELS
    channel = source_inner - head * CHANNELS

    batch = row_offsets // TOKENS
    token = row_offsets - batch * TOKENS
    source_offsets = (
        batch * STRIDE_B
        + head * STRIDE_H
        + token * STRIDE_T
        + channel * STRIDE_C
    )
    mask = (row_offsets < ROWS) & (col_offsets < COLS)

    q_vals = tl.load(q_ptr + source_offsets, mask=mask & (source == 0), other=0.0).to(tl.float32)
    k_vals = tl.load(k_ptr + source_offsets, mask=mask & (source == 1), other=0.0).to(tl.float32)
    v_vals = tl.load(v_ptr + source_offsets, mask=mask & (source == 2), other=0.0).to(tl.float32)
    values = q_vals + k_vals + v_vals

    matrix_offsets = row_offsets * COLS + col_offsets
    tl.store(matrix_ptr + matrix_offsets, values, mask=mask)

    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(
        partial_ptr + row_tile * COLS + col_offsets,
        partial[None, :],
        mask=col_offsets < COLS,
    )


@triton.jit
def _finish_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < COLS)
    values = tl.load(
        partial_ptr + tiles[:, None] * COLS + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < COLS)


def _oracle_impl(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_COLS: int,
    FINAL_BLOCK_COLS: int,
    materialize_warps: int,
    final_warps: int,
):
    q, k, v, _shape0, _shape1, _shape2, _shape3 = inputs
    batch = int(q.shape[0])
    heads = int(q.shape[1])
    tokens = int(q.shape[2])
    channels = int(q.shape[3])
    rows = batch * tokens
    cols = 3 * heads * channels

    matrix = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=q.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((cols,), device=q.device, dtype=torch.float32)
    num_row_tiles = triton.cdiv(rows, ROW_BLOCK)
    partial = torch.empty((num_row_tiles, cols), device=q.device, dtype=torch.float32)

    _materialize_partial_sum_kernel[(num_row_tiles, triton.cdiv(cols, BLOCK_COLS))](
        q,
        k,
        v,
        matrix,
        partial,
        ROWS=rows,
        COLS=cols,
        HEADS=heads,
        TOKENS=tokens,
        CHANNELS=channels,
        STRIDE_B=int(q.stride(0)),
        STRIDE_H=int(q.stride(1)),
        STRIDE_T=int(q.stride(2)),
        STRIDE_C=int(q.stride(3)),
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=materialize_warps,
        num_stages=3,
    )
    _finish_sum_kernel[(triton.cdiv(cols, FINAL_BLOCK_COLS),)](
        partial,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        COLS=cols,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=final_warps,
        num_stages=3,
    )
    return matrix, torch.as_strided(matrix, (cols, rows), (1, cols)), sum_out


# (T([128,3,197,64], bf16, stride=(37824,64,192,1)), ...)
@oracle_impl(hardware="B200", point="8a12efe4", ROW_BLOCK=128, BLOCK_COLS=64, FINAL_BLOCK_COLS=32, materialize_warps=4, final_warps=4)
# (T([128,12,198,64], bf16, stride=(152064,64,768,1)), ...)
@oracle_impl(hardware="B200", point="ae8c34ef", ROW_BLOCK=128, BLOCK_COLS=64, FINAL_BLOCK_COLS=32, materialize_warps=4, final_warps=4)
# (T([512,4,64,48], bf16, stride=(12288,48,192,1)), ...)
@oracle_impl(hardware="B200", point="7f46223b", ROW_BLOCK=128, BLOCK_COLS=64, FINAL_BLOCK_COLS=32, materialize_warps=8, final_warps=4)
# (T([128,12,1370,64], bf16, stride=(1052160,64,768,1)), ...)
@oracle_impl(hardware="B200", point="15da2150", ROW_BLOCK=128, BLOCK_COLS=64, FINAL_BLOCK_COLS=16, materialize_warps=4, final_warps=8)
# (T([128,12,256,64], bf16, stride=(196608,64,768,1)), ...)
@oracle_impl(hardware="B200", point="ac9c688a", ROW_BLOCK=128, BLOCK_COLS=64, FINAL_BLOCK_COLS=32, materialize_warps=4, final_warps=4)
def oracle_forward(inputs, **kwargs):
    return _oracle_impl(inputs, **kwargs)
