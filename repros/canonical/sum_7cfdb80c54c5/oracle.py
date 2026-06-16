"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the returned bf16 scalar-where tensor and accumulates the converted f32 channel sum from the same selected bf16 values in the producer tile, whereas Inductor splits the large channels-last points into a separate where materialization followed by generic reduction/finalization kernels that reread that tensor; Inductor cannot do this today because its scheduler does not fuse a live returned producer with a sibling split-NHW channel reduction while preserving the bf16 selected-value boundary and f32 sum-convert lowering; the fix is SCHEDULER_FUSION: teach scheduler/codegen to emit a fused materialize-plus-partial-reduction producer for returned tensors with sibling channel reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _where_partial_sum_kernel(
    mask_ptr,
    scalar_ptr,
    source_ptr,
    out_ptr,
    partial_ptr,
    sum_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    GROUP_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STORE_DIRECT_SUM: tl.constexpr,
    LOAD_SOURCE_ALWAYS: tl.constexpr,
):
    group = tl.program_id(0)
    c_block = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C
    scalar = tl.load(scalar_ptr).to(tl.float32)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.range(0, GROUP_R, BLOCK_R):
        r = group * GROUP_R + start + rows
        offsets = r[:, None] * C + cols[None, :]
        active = (r[:, None] < R) & col_mask[None, :]

        mask_value = tl.load(
            mask_ptr + offsets,
            mask=active,
            other=1.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        take_scalar = mask_value <= 0.0
        if LOAD_SOURCE_ALWAYS:
            source = tl.load(
                source_ptr + offsets,
                mask=active,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            source = tl.load(
                source_ptr + offsets,
                mask=active & ~take_scalar,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        selected = tl.where(take_scalar, scalar, source)
        selected = tl.where(active, selected, 0.0)

        tl.store(out_ptr + offsets, selected, mask=active)
        acc += tl.sum(selected, axis=0)

    if STORE_DIRECT_SUM:
        tl.store(sum_ptr + cols, acc, mask=col_mask)
    else:
        tl.store(partial_ptr + group * C + cols, acc, mask=col_mask)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_GROUPS: tl.constexpr,
    C: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    groups = tl.arange(0, BLOCK_GROUPS)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < C)
    values = tl.load(
        partial_ptr + groups[:, None] * C + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(sum_ptr + cols, total, mask=cols < C)


# (T([128,128,1,1], bf16), T([], bf16), T([128,128,1,1], bf16))
@oracle_impl(hardware="B200", point="3523706e", GROUP_R=128, BLOCK_R=128, BLOCK_C=4, FINAL_BLOCK_C=16, LOAD_SOURCE_ALWAYS=True, num_warps=4)
# (T([128,256,1,1], bf16), T([], bf16), T([128,256,1,1], bf16))
@oracle_impl(hardware="B200", point="14f2271d", GROUP_R=128, BLOCK_R=128, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,768,1,1], bf16), T([], bf16), T([128,768,1,1], bf16))
@oracle_impl(hardware="B200", point="ef334e73", GROUP_R=128, BLOCK_R=128, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
# (T([512,20,1,1], bf16), T([], bf16), T([512,20,1,1], bf16))
@oracle_impl(hardware="B200", point="c9cc6b74", GROUP_R=64, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=4)
# (T([512,32,1,1], bf16), T([], bf16), T([512,32,1,1], bf16))
@oracle_impl(hardware="B200", point="b96a53f1", GROUP_R=64, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=4)
# (T([512,120,1,1], bf16), T([], bf16), T([512,120,1,1], bf16))
@oracle_impl(hardware="B200", point="59d19ce6", GROUP_R=64, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=4)
# (T([512,168,1,1], bf16), T([], bf16), T([512,168,1,1], bf16))
@oracle_impl(hardware="B200", point="eb086b37", GROUP_R=64, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=4)
# (T([512,240,1,1], bf16), T([], bf16), T([512,240,1,1], bf16))
@oracle_impl(hardware="B200", point="d8c8090e", GROUP_R=64, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=4)
# (T([512,24,1,1], bf16), T([], bf16), T([512,24,1,1], bf16))
@oracle_impl(hardware="B200", point="f480845f", GROUP_R=64, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=4)
# (T([128,64,1,1], bf16), T([], bf16), T([128,64,1,1], bf16))
@oracle_impl(hardware="B200", point="f759a34a", GROUP_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=4)
# (T([128,384,1,1], bf16), T([], bf16), T([128,384,1,1], bf16))
@oracle_impl(hardware="B200", point="e7edaf95", GROUP_R=128, BLOCK_R=128, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,384,13,13], bf16, stride=(64896,1,4992,384)), ...)
@oracle_impl(hardware="B200", point="a20b8dac", GROUP_R=256, BLOCK_R=64, BLOCK_C=64, FINAL_BLOCK_C=16, LOAD_SOURCE_ALWAYS=True, num_warps=8)
# (T([128,256,13,13], bf16, stride=(43264,1,3328,256)), ...)
@oracle_impl(hardware="B200", point="663c0ae7", GROUP_R=128, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, LOAD_SOURCE_ALWAYS=True, num_warps=8)
# (T([32,24,1,1], bf16), T([], bf16), T([32,24,1,1], bf16))
@oracle_impl(hardware="B200", point="af287ed4", GROUP_R=32, BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=4)
# (T([32,32,1,1], bf16), T([], bf16), T([32,32,1,1], bf16))
@oracle_impl(hardware="B200", point="48c0f941", GROUP_R=32, BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=4)
# (T([32,120,1,1], bf16), T([], bf16), T([32,120,1,1], bf16))
@oracle_impl(hardware="B200", point="e2abc8eb", GROUP_R=32, BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=4)
# (T([32,168,1,1], bf16), T([], bf16), T([32,168,1,1], bf16))
@oracle_impl(hardware="B200", point="c56de08d", GROUP_R=32, BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=4)
# (T([32,240,1,1], bf16), T([], bf16), T([32,240,1,1], bf16))
@oracle_impl(hardware="B200", point="bb80badd", GROUP_R=32, BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=4)
# (T([64,64,224,224], bf16, stride=(3211264,1,14336,64)), ...)
@oracle_impl(hardware="B200", point="2d5b3c9a", GROUP_R=1024, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=8, num_warps=8)
# (T([64,128,112,112], bf16, stride=(1605632,1,14336,128)), ...)
@oracle_impl(hardware="B200", point="6b77577e", GROUP_R=1024, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=8, num_warps=8)
# (T([64,256,56,56], bf16, stride=(802816,1,14336,256)), ...)
@oracle_impl(hardware="B200", point="feea43c5", GROUP_R=1024, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
# (T([64,512,28,28], bf16, stride=(401408,1,14336,512)), ...)
@oracle_impl(hardware="B200", point="b2b247db", GROUP_R=512, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
# (T([64,512,14,14], bf16, stride=(100352,1,7168,512)), ...)
@oracle_impl(hardware="B200", point="e61b46b8", GROUP_R=512, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
def oracle_forward(
    inputs,
    *,
    GROUP_R: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
    LOAD_SOURCE_ALWAYS: bool = False,
):
    mask_input, scalar, source = inputs
    n = int(mask_input.shape[0])
    c = int(mask_input.shape[1])
    h = int(mask_input.shape[2])
    w = int(mask_input.shape[3])
    r = n * h * w

    where_out = torch.empty_strided(
        tuple(mask_input.shape),
        tuple(mask_input.stride()),
        device=mask_input.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((c,), device=mask_input.device, dtype=torch.float32)

    num_groups = triton.cdiv(r, GROUP_R)
    direct_sum = num_groups == 1
    partial = (
        sum_out
        if direct_sum
        else torch.empty((num_groups, c), device=mask_input.device, dtype=torch.float32)
    )

    _where_partial_sum_kernel[(num_groups, triton.cdiv(c, BLOCK_C))](
        mask_input,
        scalar,
        source,
        where_out,
        partial,
        sum_out,
        R=r,
        C=c,
        GROUP_R=GROUP_R,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        STORE_DIRECT_SUM=direct_sum,
        LOAD_SOURCE_ALWAYS=LOAD_SOURCE_ALWAYS,
        num_warps=num_warps,
    )

    if not direct_sum:
        _final_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
            partial,
            sum_out,
            NUM_GROUPS=num_groups,
            C=c,
            BLOCK_GROUPS=triton.next_power_of_2(num_groups),
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=8,
        )

    return where_out, sum_out
