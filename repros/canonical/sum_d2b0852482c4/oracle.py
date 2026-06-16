"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Albert/Electra three-add materialization plus column-reduction scope by sharing the same rounded producer between the returned contiguous `[M,N]` tensor, its returned transpose alias, and the bf16-rounded f32 `[N]` sum, whereas Inductor currently materializes the add/view/permute producer and schedules the compatible column reduction as separate generic work over that producer; Inductor cannot do this today because its scheduler/codegen does not fuse a mandatory returned layout backing with a sibling reduction while preserving the sequential bf16 add rounding boundaries, aliasing transpose view, and final f32-sum-to-bf16-to-f32 round trip; the fix is SCHEDULER_FUSION: add a materialize-plus-partial-reduction template that emits returned view storage and reduction partials from one producer traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add3_partial_sum_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    out_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = rows[:, None] * N + cols[None, :]

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    value = _add_rn(x1, x0).to(tl.bfloat16)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    value = _add_rn(value.to(tl.float32), x2).to(tl.bfloat16)
    x3 = tl.load(x3_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    value = _add_rn(value.to(tl.float32), x3).to(tl.bfloat16)

    tl.store(out_ptr + offsets, value, mask=mask)
    partial = tl.sum(tl.where(mask, value.to(tl.float32), 0.0), axis=0)
    tl.store(partial_ptr + row_tile * N + cols, partial, mask=cols < N)


@triton.jit
def _finish_bf16_rounded_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    N: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < N)
    offsets = tiles[:, None] * N + cols[None, :]
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(partials, axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < N)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _launch(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_COLS: int,
    FINAL_BLOCK_COLS: int,
    materialize_warps: int,
    final_warps: int,
):
    x0, x1, x2, x3, _shape0, _shape1, _shape2, out_shape, sum_shape = inputs
    m, n = _shape_tuple(out_shape)

    out = torch.empty_strided((m, n), (n, 1), device=x0.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(
        _shape_tuple(sum_shape),
        (1,),
        device=x0.device,
        dtype=torch.float32,
    )
    num_row_tiles = triton.cdiv(m, ROW_BLOCK)
    partial = torch.empty_strided(
        (num_row_tiles, n),
        (n, 1),
        device=x0.device,
        dtype=torch.float32,
    )

    _add3_partial_sum_kernel[(num_row_tiles, triton.cdiv(n, BLOCK_COLS))](
        x0,
        x1,
        x2,
        x3,
        out,
        partial,
        M=m,
        N=n,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=materialize_warps,
        num_stages=3,
    )
    _finish_bf16_rounded_sum_kernel[(triton.cdiv(n, FINAL_BLOCK_COLS),)](
        partial,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        N=n,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=final_warps,
        num_stages=3,
    )
    return out, torch.as_strided(out, (n, m), (1, n)), sum_out


# d247a9e9: (T([4096,4096], bf16), T([8,512,4096], bf16), ...)
@oracle_impl(hardware="B200", point="d247a9e9", ROW_BLOCK=128, BLOCK_COLS=64, FINAL_BLOCK_COLS=64, materialize_warps=8, final_warps=4)
# 1b39f873: (T([32768,256], bf16), T([64,512,256], bf16), ...)
@oracle_impl(hardware="B200", point="1b39f873", ROW_BLOCK=256, BLOCK_COLS=32, FINAL_BLOCK_COLS=32, materialize_warps=8, final_warps=8)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_COLS: int,
    FINAL_BLOCK_COLS: int,
    materialize_warps: int,
    final_warps: int,
):
    return _launch(
        inputs,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_COLS=BLOCK_COLS,
        FINAL_BLOCK_COLS=FINAL_BLOCK_COLS,
        materialize_warps=materialize_warps,
        final_warps=final_warps,
    )
