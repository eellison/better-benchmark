"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete max-pool-backward scatter, bf16 mask overwrite, returned channels-last `where` tensor, and sibling bf16 channel sum by reverse-gathering each structured scatter destination directly into the materialized output while accumulating per-channel partials, whereas Inductor materializes a dense f32 zero/scatter_add buffer, casts it to bf16, runs the mask overwrite, and then rereads the returned tensor through a separate reduction; Inductor cannot do this today because its scheduler/codegen treats `_low_memory_max_pool_offsets_to_indices` plus `scatter_add` as an opaque scatter producer rather than a fixed-window maxpool-backward scatter-reduce with a live materialized side output; the fix is SCATTER_REDUCE: add a guarded low-memory maxpool-backward scatter-reduce lowering that emits the bf16 output store and dependent channel reduction from one destination-space traversal without the f32 scatter intermediate."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _pool_scatter_where_partials_kernel(
    grad_ptr,
    offsets_ptr,
    mask_ptr,
    scalar_ptr,
    where_out_ptr,
    partial_ptr,
    sum_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    KERNEL: tl.constexpr,
    GROUP_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STORE_DIRECT_SUM: tl.constexpr,
    CONST_OFFSET_3: tl.constexpr,
):
    group = tl.program_id(0)
    c_block = tl.program_id(1)
    rows_base = tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C
    scalar = tl.load(scalar_ptr)
    acc_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.range(0, GROUP_R, BLOCK_R):
        rows = group * GROUP_R + start + rows_base
        row_active = rows < R
        n = rows // (IH * IW)
        rem = rows - n * (IH * IW)
        ih = rem // IW
        iw = rem - ih * IW
        out_offsets = rows[:, None] * C + cols[None, :]
        active = row_active[:, None] & col_mask[None, :]

        mask_value = tl.load(mask_ptr + out_offsets, mask=active, other=1)
        take_scalar = mask_value != 0
        need_scatter = active & (~take_scalar)
        scatter = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

        if CONST_OFFSET_3:
            src_oh_num = ih - 1
            src_ow_num = iw - 1
            valid_src_row = (
                row_active
                & (src_oh_num >= 0)
                & (src_ow_num >= 0)
                & ((src_oh_num % 2) == 0)
                & ((src_ow_num % 2) == 0)
            )
            src_oh = src_oh_num // 2
            src_ow = src_ow_num // 2
            valid_src_row = valid_src_row & (src_oh < OH) & (src_ow < OW)
            src_rows = (n * OH + src_oh) * OW + src_ow
            src_offsets = src_rows[:, None] * C + cols[None, :]
            src_active = valid_src_row[:, None] & need_scatter
            scatter += tl.load(
                grad_ptr + src_offsets,
                mask=src_active,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            for ky in tl.static_range(0, 3):
                for kx in tl.static_range(0, 3):
                    src_oh_num = ih - ky
                    src_ow_num = iw - kx
                    valid_src_row = (
                        row_active
                        & (ky < KERNEL)
                        & (kx < KERNEL)
                        & (src_oh_num >= 0)
                        & (src_ow_num >= 0)
                        & ((src_oh_num % 2) == 0)
                        & ((src_ow_num % 2) == 0)
                    )
                    src_oh = src_oh_num // 2
                    src_ow = src_ow_num // 2
                    valid_src_row = valid_src_row & (src_oh < OH) & (src_ow < OW)
                    src_rows = (n * OH + src_oh) * OW + src_ow
                    src_offsets = src_rows[:, None] * C + cols[None, :]
                    src_active = valid_src_row[:, None] & need_scatter
                    pool_offset = tl.load(
                        offsets_ptr + src_offsets,
                        mask=src_active,
                        other=-1,
                        eviction_policy="evict_first",
                    ).to(tl.int32)
                    matches = pool_offset == (ky * KERNEL + kx)
                    scatter += tl.load(
                        grad_ptr + src_offsets,
                        mask=src_active & matches,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)

        scatter_bf16 = scatter.to(tl.bfloat16, fp_downcast_rounding="rtne")
        selected = tl.where(take_scalar, scalar, scatter_bf16)
        tl.store(where_out_ptr + out_offsets, selected, mask=active)
        acc_sum += tl.sum(tl.where(active, selected.to(tl.float32), 0.0), axis=0)

    if STORE_DIRECT_SUM:
        tl.store(
            sum_ptr + cols,
            acc_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
            mask=col_mask,
        )
    else:
        tl.store(partial_ptr + group * C + cols, acc_sum, mask=col_mask)


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
    partials = tl.load(
        partial_ptr + groups[:, None] * C + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(partials, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < C)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _launch(
    inputs,
    *,
    KERNEL: int,
    GROUP_R: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    CONST_OFFSET_3: bool = False,
    num_warps: int,
):
    grad, offsets, mask, scalar, *_shape_params = inputs
    n = int(grad.shape[0])
    c = int(grad.shape[1])
    oh = int(grad.shape[2])
    ow = int(grad.shape[3])
    ih = int(mask.shape[2])
    iw = int(mask.shape[3])
    r = n * ih * iw

    where_out = torch.empty_strided(
        tuple(mask.shape),
        tuple(int(stride) for stride in mask.stride()),
        device=mask.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((c,), device=mask.device, dtype=torch.float32)
    num_groups = triton.cdiv(r, GROUP_R)
    direct_sum = num_groups == 1
    partial = (
        sum_out
        if direct_sum
        else torch.empty((num_groups, c), device=mask.device, dtype=torch.float32)
    )

    _pool_scatter_where_partials_kernel[(num_groups, triton.cdiv(c, BLOCK_C))](
        grad,
        offsets,
        mask,
        scalar,
        where_out,
        partial,
        sum_out,
        R=r,
        C=c,
        OH=oh,
        OW=ow,
        IH=ih,
        IW=iw,
        KERNEL=KERNEL,
        GROUP_R=GROUP_R,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        STORE_DIRECT_SUM=direct_sum,
        CONST_OFFSET_3=CONST_OFFSET_3,
        num_warps=num_warps,
        num_stages=3,
    )
    if not direct_sum:
        _final_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
            partial,
            sum_out,
            NUM_GROUPS=num_groups,
            C=c,
            BLOCK_GROUPS=_next_power_of_2(num_groups),
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=8,
            num_stages=3,
        )

    return where_out, sum_out


# (T([128,64,27,27], bf16, stride=(46656,1,1728,64)), T([128,64,27,27], i8, stride=(46656,1,1728,64)), T([128,64,55,55], b8, stride=(193600,1,3520,64)), T([], bf16))
@oracle_impl(hardware="B200", point="94a62ed8", KERNEL=3, GROUP_R=128, BLOCK_R=16, BLOCK_C=16, FINAL_BLOCK_C=8, num_warps=4)
# (T([128,192,13,13], bf16, stride=(32448,1,2496,192)), T([128,192,13,13], i8, stride=(32448,1,2496,192)), T([128,192,27,27], b8, stride=(139968,1,5184,192)), T([], bf16))
@oracle_impl(hardware="B200", point="e579fe5c", KERNEL=3, GROUP_R=128, BLOCK_R=16, BLOCK_C=16, FINAL_BLOCK_C=8, num_warps=4)
# (T([32,64,55,55], bf16, stride=(193600,1,3520,64)), T([32,64,55,55], i8, stride=(193600,1,3520,64)), T([32,64,111,111], b8, stride=(788544,1,7104,64)), T([], bf16))
@oracle_impl(hardware="B200", point="a00ef5f5", KERNEL=3, GROUP_R=128, BLOCK_R=16, BLOCK_C=16, FINAL_BLOCK_C=8, num_warps=4)
# (T([64,64,112,112], bf16, stride=(802816,1,7168,64)), T([64,64,112,112], i8, stride=(802816,1,7168,64), gen=const(3)), T([64,64,224,224], b8, stride=(3211264,1,14336,64)), T([], bf16))
@oracle_impl(hardware="B200", point="b721890d", KERNEL=2, GROUP_R=1024, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=8, CONST_OFFSET_3=True, num_warps=8)
# (T([64,128,56,56], bf16, stride=(401408,1,7168,128)), T([64,128,56,56], i8, stride=(401408,1,7168,128), gen=const(3)), T([64,128,112,112], b8, stride=(1605632,1,14336,128)), T([], bf16))
@oracle_impl(hardware="B200", point="c0a53b72", KERNEL=2, GROUP_R=1024, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=8, CONST_OFFSET_3=True, num_warps=8)
# (T([64,256,28,28], bf16, stride=(200704,1,7168,256)), T([64,256,28,28], i8, stride=(200704,1,7168,256), gen=const(3)), T([64,256,56,56], b8, stride=(802816,1,14336,256)), T([], bf16))
@oracle_impl(hardware="B200", point="b3ea8ee2", KERNEL=2, GROUP_R=1024, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, CONST_OFFSET_3=True, num_warps=8)
# (T([64,512,14,14], bf16, stride=(100352,1,7168,512)), T([64,512,14,14], i8, stride=(100352,1,7168,512), gen=const(3)), T([64,512,28,28], b8, stride=(401408,1,14336,512)), T([], bf16))
@oracle_impl(hardware="B200", point="bf7c8e5d", KERNEL=2, GROUP_R=512, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, CONST_OFFSET_3=True, num_warps=8)
def oracle_forward(
    inputs,
    *,
    KERNEL: int,
    GROUP_R: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
    CONST_OFFSET_3: bool = False,
):
    return _launch(
        inputs,
        KERNEL=KERNEL,
        GROUP_R=GROUP_R,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        CONST_OFFSET_3=CONST_OFFSET_3,
        num_warps=num_warps,
    )
