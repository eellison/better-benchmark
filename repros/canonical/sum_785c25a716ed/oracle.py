"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle writes the bf16 contiguous clone backing for both returned layout views, fuses clone plus partial-sum for contiguous-stride points, and uses a stride-specialized H-last clone before the same partial/final reduction for the non-contiguous point while preserving the required bf16 round-trip on the sum output, whereas Inductor emits a materializing layout clone and then separate reduction kernels that reread the cloned buffer; Inductor cannot do this today because its scheduler does not fuse a live materialized layout-copy output with a sibling reduction over the same producer iteration space while preserving aliasing view returns; the fix is SCHEDULER_FUSION: add a multi-output layout-copy-plus-reduction schedule that emits the returned clone storage, alias-only transposed view, and reduction partials from one producer traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _copy_partial_sum_kernel(
    x_ptr,
    clone_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)

    rows = row_tile * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    out_offsets = rows[:, None] * COLS + cols[None, :]
    values = tl.load(x_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(clone_ptr + out_offsets, values, mask=mask)

    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(partial_ptr + row_tile * COLS + cols, partial, mask=cols < COLS)


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
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < COLS)


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


def _oracle_impl(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_COLS: int,
    FINAL_BLOCK_COLS: int,
    FINAL_BLOCK_COLS_HLAST: int,
    COPY_Y_BLOCK: int,
    COPY_X_BLOCK: int,
    REDUCE_X_BLOCK: int,
    copy_warps: int,
    fused_warps: int,
    reduce_warps: int,
    final_warps: int,
):
    x, _shape_param_0, shape_2d, sum_shape = inputs
    rows = int(shape_2d[0])
    cols = int(shape_2d[1])
    seq = int(x.shape[2])
    head_dim = int(x.shape[3])
    contiguous_permute = (
        int(x.stride(0)) == seq * cols
        and int(x.stride(1)) == head_dim
        and int(x.stride(2)) == cols
        and int(x.stride(3)) == 1
    )

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

    final_block_cols = FINAL_BLOCK_COLS if contiguous_permute else FINAL_BLOCK_COLS_HLAST

    num_row_tiles = triton.cdiv(rows, ROW_BLOCK)
    partial = torch.empty_strided(
        (num_row_tiles, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.float32,
    )

    if contiguous_permute:
        _copy_partial_sum_kernel[(num_row_tiles, triton.cdiv(cols, BLOCK_COLS))](
            x,
            clone,
            partial,
            ROWS=rows,
            COLS=cols,
            ROW_BLOCK=ROW_BLOCK,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=fused_warps,
            num_stages=3,
        )
    else:
        heads = cols // head_dim
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
    _finish_sum_kernel[(triton.cdiv(cols, final_block_cols),)](
        partial,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        COLS=cols,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_COLS=final_block_cols,
        num_warps=final_warps,
        num_stages=3,
    )
    return clone, torch.as_strided(clone, (cols, rows), (1, cols)), sum_out


# (T([16,16,512,64], bf16, stride=(524288,64,1024,1)), S([16,512,1024]), S([8192,1024]), S([1024]))
@oracle_impl(
    hardware="B200",
    point="903ae292",
    ROW_BLOCK=512,
    BLOCK_COLS=32,
    FINAL_BLOCK_COLS=32,
    FINAL_BLOCK_COLS_HLAST=32,
    COPY_Y_BLOCK=16,
    COPY_X_BLOCK=64,
    REDUCE_X_BLOCK=32,
    copy_warps=4,
    fused_warps=8,
    reduce_warps=4,
    final_warps=4,
)
# (T([32,6,512,64], bf16, stride=(196608,1,384,6)), S([32,512,384]), S([16384,384]), S([384]))
@oracle_impl(
    hardware="B200",
    point="f85fe078",
    ROW_BLOCK=512,
    BLOCK_COLS=32,
    FINAL_BLOCK_COLS=32,
    FINAL_BLOCK_COLS_HLAST=32,
    COPY_Y_BLOCK=16,
    COPY_X_BLOCK=64,
    REDUCE_X_BLOCK=32,
    copy_warps=4,
    fused_warps=8,
    reduce_warps=4,
    final_warps=4,
)
# (T([32,6,512,64], bf16, stride=(196608,64,384,1)), S([32,512,384]), S([16384,384]), S([384]))
@oracle_impl(
    hardware="B200",
    point="9876fbcf",
    ROW_BLOCK=512,
    BLOCK_COLS=32,
    FINAL_BLOCK_COLS=32,
    FINAL_BLOCK_COLS_HLAST=32,
    COPY_Y_BLOCK=16,
    COPY_X_BLOCK=64,
    REDUCE_X_BLOCK=32,
    copy_warps=4,
    fused_warps=8,
    reduce_warps=4,
    final_warps=4,
)
def oracle_forward(inputs, **kwargs):
    return _oracle_impl(inputs, **kwargs)
