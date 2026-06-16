"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete M2M100 bf16 masked materialization plus sibling column-sum scope by writing the `where(mask, 0, view(arg0))` producer once into the required hidden `[64,128,4096]` backing storage, returning both the flat `[8192,4096]` view and its metadata-only `[4096,8192]` transpose alias, and accumulating the dim-0 fp32 column sum from those bf16-rounded producer values before applying the captured final bf16-to-f32 round-trip, whereas Inductor lowers the view/where/view/permute/sum/cast chain through generic multi-output reduction scheduling with avoidable producer replay and materialization overhead; Inductor cannot do this today because its scheduler does not form an alias-aware full-scope materialize-plus-column-reduction plan for a bf16 producer that is both returned and reduced; the fix is SCHEDULER_FUSION: add a reduction schedule that keeps the masked producer virtual, emits the visible backing storage once, returns view aliases, and finalizes compatible column reductions from the same traversal while preserving explicit bf16 rounding boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _masked_materialize_partial_sum_kernel(
    x_ptr,
    mask_ptr,
    backing_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]

    x = tl.load(x_ptr + offsets)
    mask_values = tl.load(mask_ptr + offsets)
    values = tl.where(mask_values != 0, 0.0, x).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(backing_ptr + offsets, values)
    partial = tl.sum(values.to(tl.float32), axis=0)
    tl.store(partial_ptr + row_tile * N + cols, partial)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    N: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tiles = tl.arange(0, BLOCK_TILES)
    partials = tl.load(
        partial_ptr + tiles[:, None] * N + cols[None, :],
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(partials, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, rounded)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# (T([8192,4096], bf16), T([64,128,4096], b8), S([64,128,4096]), S([8192,4096]), S([4096]))
@oracle_impl(
    hardware="B200",
    point="4a8b763e",
    ROW_BLOCK=128,
    BLOCK_N=128,
    FINAL_BLOCK_N=32,
    producer_warps=8,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_N: int,
    FINAL_BLOCK_N: int,
    producer_warps: int,
    final_warps: int,
):
    x, mask, shape3d_arg, flat_shape_arg, sum_shape_arg = inputs
    shape3d = _shape_tuple(shape3d_arg)
    flat_shape = _shape_tuple(flat_shape_arg)
    sum_shape = _shape_tuple(sum_shape_arg)
    m = int(flat_shape[0])
    n = int(flat_shape[1])
    num_row_tiles = triton.cdiv(m, ROW_BLOCK)

    backing = torch.empty_strided(
        shape3d,
        (shape3d[1] * shape3d[2], shape3d[2], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (num_row_tiles, n),
        (n, 1),
        device=x.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        sum_shape,
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    _masked_materialize_partial_sum_kernel[(num_row_tiles, triton.cdiv(n, BLOCK_N))](
        x,
        mask,
        backing,
        partial,
        M=m,
        N=n,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_N=BLOCK_N,
        num_warps=producer_warps,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(n, FINAL_BLOCK_N),)](
        partial,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        N=n,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=final_warps,
        num_stages=3,
    )

    flat = torch.as_strided(backing, flat_shape, (n, 1))
    return flat, torch.as_strided(backing, (n, m), (1, n)), sum_out
