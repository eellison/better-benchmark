"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the returned Demucs bf16 scalar-where tensor and accumulates Inductor's converted f32 channel sum from the same selected bf16 values in BCT source-layout tiles, whereas Inductor schedules the observable where producer and the sibling `[0, 2]` reduction as separate generic regions that reread or reprocess the same large contiguous tensor. Inductor cannot do this today because its scheduler does not fuse a live returned producer with a sibling channel reduction while preserving the bf16 selected-value boundary, the f32 reduction output, and the contiguous `[B, C, T]` output layout; the fix is SCHEDULER_FUSION: teach scheduler/codegen to emit a fused materialize-plus-partial-reduction producer for returned tensors with sibling channel reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4


@triton.jit
def _where_partial_sum_bct_kernel(
    mask_ptr,
    scalar_ptr,
    source_ptr,
    where_ptr,
    partial_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    TILES_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
    LOAD_SOURCE_ALWAYS: tl.constexpr,
):
    c_block = tl.program_id(0)
    batch = tl.program_id(1)
    tile_t = tl.program_id(2)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    ts = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    active = (cols[:, None] < C) & (ts[None, :] < T)
    offsets = batch * C * T + cols[:, None] * T + ts[None, :]

    mask_value = tl.load(
        mask_ptr + offsets,
        mask=active,
        other=1.0,
        eviction_policy="evict_first",
    )
    take_scalar = mask_value <= 0.0
    scalar = tl.load(scalar_ptr)
    if LOAD_SOURCE_ALWAYS:
        source = tl.load(
            source_ptr + offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        )
    else:
        source = tl.load(
            source_ptr + offsets,
            mask=active & ~take_scalar,
            other=0.0,
            eviction_policy="evict_first",
        )
    selected = tl.where(take_scalar, scalar, source)
    sum_values = tl.where(active, selected.to(tl.float32), 0.0)

    tl.store(where_ptr + offsets, selected, mask=active)
    partial = tl.sum(sum_values, axis=1)
    group = batch * TILES_T + tile_t
    tl.store(partial_ptr + group * C + cols, partial, mask=cols < C)


@triton.jit
def _zero_vector_kernel(out_ptr, C: tl.constexpr, BLOCK_C: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK_C,), dtype=tl.float32), mask=offsets < C)


@triton.jit
def _where_atomic_sum_bct_kernel(
    mask_ptr,
    scalar_ptr,
    source_ptr,
    where_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
    LOAD_SOURCE_ALWAYS: tl.constexpr,
):
    c_block = tl.program_id(0)
    batch = tl.program_id(1)
    tile_t = tl.program_id(2)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    ts = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    active = (cols[:, None] < C) & (ts[None, :] < T)
    offsets = batch * C * T + cols[:, None] * T + ts[None, :]

    mask_value = tl.load(
        mask_ptr + offsets,
        mask=active,
        other=1.0,
        eviction_policy="evict_first",
    )
    take_scalar = mask_value <= 0.0
    scalar = tl.load(scalar_ptr)
    if LOAD_SOURCE_ALWAYS:
        source = tl.load(
            source_ptr + offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        )
    else:
        source = tl.load(
            source_ptr + offsets,
            mask=active & ~take_scalar,
            other=0.0,
            eviction_policy="evict_first",
        )
    selected = tl.where(take_scalar, scalar, source)
    sum_values = tl.where(active, selected.to(tl.float32), 0.0)

    tl.store(where_ptr + offsets, selected, mask=active)
    partial = tl.sum(sum_values, axis=1)
    tl.atomic_add(out_ptr + cols, partial, sem="relaxed", mask=cols < C)


@triton.jit
def _where_batch_atomic_sum_bct_kernel(
    mask_ptr,
    scalar_ptr,
    source_ptr,
    where_ptr,
    out_ptr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    LOAD_SOURCE_ALWAYS: tl.constexpr,
):
    channel = tl.program_id(0)
    batch = tl.program_id(1)
    t = tl.arange(0, BLOCK_T)
    active = t < T
    offsets = batch * C * T + channel * T + t

    mask_value = tl.load(
        mask_ptr + offsets,
        mask=active,
        other=1.0,
        eviction_policy="evict_first",
    )
    take_scalar = mask_value <= 0.0
    scalar = tl.load(scalar_ptr)
    if LOAD_SOURCE_ALWAYS:
        source = tl.load(
            source_ptr + offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        )
    else:
        source = tl.load(
            source_ptr + offsets,
            mask=active & ~take_scalar,
            other=0.0,
            eviction_policy="evict_first",
        )
    selected = tl.where(take_scalar, scalar, source)
    sum_values = tl.where(active, selected.to(tl.float32), 0.0)

    tl.store(where_ptr + offsets, selected, mask=active)
    tl.atomic_add(out_ptr + channel, tl.sum(sum_values, axis=0), sem="relaxed")


@triton.jit
def _where_sum_channel_bct_kernel(
    mask_ptr,
    scalar_ptr,
    source_ptr,
    where_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOAD_SOURCE_ALWAYS: tl.constexpr,
):
    channel = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    active = r < (B * T)
    batch = r // T
    t = r - batch * T
    offsets = batch * C * T + channel * T + t

    mask_value = tl.load(
        mask_ptr + offsets,
        mask=active,
        other=1.0,
        eviction_policy="evict_first",
    )
    take_scalar = mask_value <= 0.0
    scalar = tl.load(scalar_ptr)
    if LOAD_SOURCE_ALWAYS:
        source = tl.load(
            source_ptr + offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        )
    else:
        source = tl.load(
            source_ptr + offsets,
            mask=active & ~take_scalar,
            other=0.0,
            eviction_policy="evict_first",
        )
    selected = tl.where(take_scalar, scalar, source)
    sum_values = tl.where(active, selected.to(tl.float32), 0.0)
    tl.store(where_ptr + offsets, selected, mask=active)
    tl.store(out_ptr + channel, tl.sum(sum_values, axis=0))


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    out_ptr,
    NUM_GROUPS: tl.constexpr,
    C: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    groups = tl.arange(0, BLOCK_GROUPS)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < C)
    values = tl.load(
        partial_ptr + groups[:, None] * C + cols[None, :],
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(out_ptr + cols, total, mask=cols < C)


@oracle_impl(hardware="B200", point="7865ae61", BLOCK_C=2, BLOCK_T=4096, FINAL_BLOCK_C=16, DIRECT=False, ATOMIC=False, BATCH_DIRECT=False, BLOCK_R=1, LOAD_SOURCE_ALWAYS=False, num_warps=8)
@oracle_impl(hardware="B200", point="c6e644f9", BLOCK_C=1, BLOCK_T=32768, FINAL_BLOCK_C=16, DIRECT=False, ATOMIC=True, BATCH_DIRECT=True, BLOCK_R=1, LOAD_SOURCE_ALWAYS=True, num_warps=16)
@oracle_impl(hardware="B200", point="9d21ed2c", BLOCK_C=1, BLOCK_T=1024, FINAL_BLOCK_C=16, DIRECT=False, ATOMIC=True, BATCH_DIRECT=False, BLOCK_R=1, LOAD_SOURCE_ALWAYS=True, num_warps=4)
@oracle_impl(hardware="B200", point="03da249b", BLOCK_C=1, BLOCK_T=1, FINAL_BLOCK_C=1, DIRECT=True, ATOMIC=False, BATCH_DIRECT=False, BLOCK_R=8192, LOAD_SOURCE_ALWAYS=False, num_warps=8)
@oracle_impl(hardware="B200", point="11e513cb", BLOCK_C=1, BLOCK_T=1, FINAL_BLOCK_C=1, DIRECT=True, ATOMIC=False, BATCH_DIRECT=False, BLOCK_R=2048, LOAD_SOURCE_ALWAYS=False, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_T: int,
    FINAL_BLOCK_C: int,
    DIRECT: bool,
    ATOMIC: bool,
    BATCH_DIRECT: bool,
    BLOCK_R: int,
    LOAD_SOURCE_ALWAYS: bool,
    num_warps: int,
):
    mask_input, scalar, source = inputs
    b = int(mask_input.shape[0])
    c = int(mask_input.shape[1])
    t = int(mask_input.shape[2])
    device = mask_input.device

    where = torch.empty_strided(
        (b, c, t),
        (c * t, t, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)

    if DIRECT:
        _where_sum_channel_bct_kernel[(c,)](
            mask_input,
            scalar,
            source,
            where,
            out,
            B=b,
            C=c,
            T=t,
            BLOCK_R=BLOCK_R,
            LOAD_SOURCE_ALWAYS=LOAD_SOURCE_ALWAYS,
            num_warps=num_warps,
            num_stages=3,
        )
        return where, out

    tiles_t = triton.cdiv(t, BLOCK_T)
    if ATOMIC:
        _zero_vector_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
            out,
            C=c,
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=4,
            num_stages=3,
        )
        if BATCH_DIRECT:
            _where_batch_atomic_sum_bct_kernel[(c, b)](
                mask_input,
                scalar,
                source,
                where,
                out,
                C=c,
                T=t,
                BLOCK_T=BLOCK_T,
                LOAD_SOURCE_ALWAYS=LOAD_SOURCE_ALWAYS,
                num_warps=num_warps,
                num_stages=3,
            )
            return where, out

        _where_atomic_sum_bct_kernel[(triton.cdiv(c, BLOCK_C), b, tiles_t)](
            mask_input,
            scalar,
            source,
            where,
            out,
            B=b,
            C=c,
            T=t,
            BLOCK_C=BLOCK_C,
            BLOCK_T=BLOCK_T,
            LOAD_SOURCE_ALWAYS=LOAD_SOURCE_ALWAYS,
            num_warps=num_warps,
            num_stages=3,
        )
        return where, out

    num_groups = b * tiles_t
    partial = torch.empty_strided(
        (num_groups, c),
        (c, 1),
        device=device,
        dtype=torch.float32,
    )

    _where_partial_sum_bct_kernel[(triton.cdiv(c, BLOCK_C), b, tiles_t)](
        mask_input,
        scalar,
        source,
        where,
        partial,
        B=BATCH,
        C=c,
        T=t,
        TILES_T=tiles_t,
        BLOCK_C=BLOCK_C,
        BLOCK_T=BLOCK_T,
        LOAD_SOURCE_ALWAYS=LOAD_SOURCE_ALWAYS,
        num_warps=num_warps,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partial,
        out,
        NUM_GROUPS=num_groups,
        C=c,
        BLOCK_GROUPS=triton.next_power_of_2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )

    return where, out
