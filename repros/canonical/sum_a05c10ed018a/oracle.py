"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the
complete Demucs bf16 zero-padded slice_scatter, materialized masked where tensor,
and compiled-path f32 channel sum in one source-space traversal, whereas
Inductor schedules the zero-fill/slice_scatter side output, where
materialization, and sibling reduction as separate generic regions over the same
contiguous source and mask. Inductor cannot do this today because its scheduler
does not model a required materialized slice_scatter side output plus an
unpadded masked reduction epilogue as one structured scatter-reduce template.
The fix is SCATTER_REDUCE: emit the padded side-output stores and accumulate the
sibling channel reduction directly from the scattered source tile."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 1024
IN_T = 364
PAD = 4
OUT_T = 372
REDUCTION = BATCH * IN_T
BLOCK_R = triton.next_power_of_2(REDUCTION)
ZERO_BLOCK = BATCH * 2 * PAD


@triton.jit
def _slice_scatter_where_sum_kernel(
    src_ptr,
    mask_ptr,
    scalar_ptr,
    padded_ptr,
    where_ptr,
    sum_ptr,
    BLOCK_R_: tl.constexpr,
    ZERO_BLOCK_: tl.constexpr,
):
    channel = tl.program_id(0)

    r = tl.arange(0, BLOCK_R_)
    active = r < 1456
    batch = r // 364
    t = r - batch * 364

    src_offsets = batch * (1024 * 364) + channel * 364 + t
    padded_offsets = batch * (1024 * 372) + channel * 372 + 4 + t

    src_vals = tl.load(src_ptr + src_offsets, mask=active, other=0.0)
    mask_vals = tl.load(mask_ptr + src_offsets, mask=active, other=0)
    scalar = tl.load(scalar_ptr)
    where_vals = tl.where(mask_vals, scalar, src_vals)

    tl.store(padded_ptr + padded_offsets, src_vals, mask=active)
    tl.store(where_ptr + src_offsets, where_vals, mask=active)

    z = tl.arange(0, ZERO_BLOCK_)
    z_batch = z // 8
    z_pos = z - z_batch * 8
    z_t = tl.where(z_pos < 4, z_pos, 372 - 4 + (z_pos - 4))
    z_offsets = z_batch * (1024 * 372) + channel * 372 + z_t
    zero = tl.full((ZERO_BLOCK_,), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(padded_ptr + z_offsets, zero)

    total = tl.sum(tl.where(active, where_vals.to(tl.float32), 0.0), axis=0)
    tl.store(sum_ptr + channel, total)


# torchbench_demucs train, bf16 [4,1024,364] padded to [4,1024,372].
@oracle_impl(hardware="B200", point="fea3ee9c", BLOCK_R=BLOCK_R, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
    src, mask, scalar, shape_param = inputs
    device = src.device

    slice_scatter = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (CHANNELS * OUT_T, OUT_T, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    where = torch.empty_strided(
        (BATCH, CHANNELS, IN_T),
        (CHANNELS * IN_T, IN_T, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    reduced = torch.empty_strided(
        (CHANNELS,),
        (1,),
        device=device,
        dtype=torch.float32,
    )

    _slice_scatter_where_sum_kernel[(CHANNELS,)](
        src,
        mask,
        scalar,
        slice_scatter,
        where,
        reduced,
        BLOCK_R_=BLOCK_R,
        ZERO_BLOCK_=ZERO_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return slice_scatter, where, reduced
