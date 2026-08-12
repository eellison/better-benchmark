"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle writes the returned bf16 contiguous clone backing for the H-last ConvBert attention layout and computes the f32 column sum with the required bf16 round-trip on the sum output, whereas Inductor materializes the permute/reshape clone chain and then schedules separate reduction kernels that reread the cloned buffer; Inductor cannot do this today because its scheduler does not fuse a live layout-copy output with a sibling reduction while preserving aliasing view returns and explicit dtype boundaries; the fix is SCHEDULER_FUSION: add a multi-output layout-copy-plus-reduction schedule that emits the returned clone storage, alias-only transposed view, and rounded reduction from one producer traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _copy_hlast_kernel(
    x_ptr,
    clone_ptr,
    YNUMEL: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COLS: tl.constexpr,
    Y_BLOCK: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    y = tl.program_id(1) * Y_BLOCK + tl.arange(0, Y_BLOCK)[:, None]
    dim = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[None, :]
    mask = (y < YNUMEL) & (dim < HEAD_DIM)

    head = y % HEADS
    row = y // HEADS
    in_offsets = row * COLS + head + dim * HEADS
    out_offsets = row * COLS + head * HEAD_DIM + dim
    values = tl.load(x_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(clone_ptr + out_offsets, values, mask=mask)


@triton.jit
def _partial_sum_from_clone_kernel(
    clone_ptr,
    partial_ptr,
    X_NUMEL: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    xidx = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)
    r = tl.arange(0, ROW_BLOCK)
    cols = xidx % COLS
    row_tiles = xidx // COLS
    rows = row_tiles[:, None] * ROW_BLOCK + r[None, :]
    mask = (xidx[:, None] < X_NUMEL) & (rows < ROWS)
    values = tl.load(
        clone_ptr + rows * COLS + cols[:, None],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    partial = tl.sum(tl.where(mask, values, 0.0), axis=1)
    tl.store(partial_ptr + xidx, partial, mask=xidx < X_NUMEL)


@triton.jit
def _finish_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col_tile = tl.program_id(0)
    cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (cols[:, None] < COLS) & (tiles[None, :] < NUM_ROW_TILES)
    values = tl.load(
        partial_ptr + tiles[None, :] * COLS + cols[:, None],
        mask=mask,
        other=0.0,
    )
    total = tl.sum(tl.where(mask, values, 0.0), axis=1)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < COLS)


# f85fe078: (T([32,6,512,64], bf16, stride=(196608,1,384,6)), S([32,512,384]), S([16384,384]), S([384]))
@oracle_impl(
    hardware="B200",
    point="f85fe078",
    ROW_BLOCK=512,
    FINAL_BLOCK_COLS=32,
    COPY_Y_BLOCK=16,
    COPY_X_BLOCK=64,
    REDUCE_X_BLOCK=32,
    copy_warps=4,
    reduce_warps=4,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    FINAL_BLOCK_COLS: int,
    COPY_Y_BLOCK: int,
    COPY_X_BLOCK: int,
    REDUCE_X_BLOCK: int,
    copy_warps: int,
    reduce_warps: int,
    final_warps: int,
):
    x, _shape_param_0, shape_2d, sum_shape = inputs
    del _shape_param_0

    rows = int(shape_2d[0])
    cols = int(shape_2d[1])
    heads = int(x.shape[1])
    head_dim = int(x.shape[3])

    clone = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in sum_shape),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    num_row_tiles = triton.cdiv(rows, ROW_BLOCK)
    partial = torch.empty_strided(
        (num_row_tiles, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _copy_hlast_kernel[
        (triton.cdiv(head_dim, COPY_X_BLOCK), triton.cdiv(rows * heads, COPY_Y_BLOCK))
    ](
        x,
        clone,
        YNUMEL=rows * heads,
        HEADS=heads,
        HEAD_DIM=head_dim,
        COLS=cols,
        Y_BLOCK=COPY_Y_BLOCK,
        X_BLOCK=COPY_X_BLOCK,
        num_warps=copy_warps,
        num_stages=3,
    )
    _partial_sum_from_clone_kernel[(triton.cdiv(num_row_tiles * cols, REDUCE_X_BLOCK),)](
        clone,
        partial,
        X_NUMEL=num_row_tiles * cols,
        ROWS=rows,
        COLS=cols,
        ROW_BLOCK=ROW_BLOCK,
        X_BLOCK=REDUCE_X_BLOCK,
        num_warps=reduce_warps,
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
    return clone, torch.as_strided(clone, (cols, rows), (1, cols)), sum_out
