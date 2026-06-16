"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Demucs bf16 zero-padded slice_scatter side output, materialized masked where tensor, and compiled-path f32 channel sum from one source-space traversal, whereas Inductor schedules the zero-fill/slice_scatter output, the strided-mask where materialization, and the sibling reduction as separate generic regions that reread the same source and mask; Inductor cannot do this today because its scheduler/codegen has no structured scatter-reduce template that keeps a required materialized zero-padded slice_scatter side output and an unpadded masked reduction epilogue fused while preserving the non-contiguous mask strides and returned materializations; the fix is SCATTER_REDUCE: add a structured slice_scatter lowering that emits padded side-output stores and accumulates sibling source-space reductions directly from the scattered tile."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 512
IN_T = 1452
PAD_LEFT = 20
PAD_RIGHT = 21
OUT_T = 1493
MASK_STRIDE_B = 786432
MASK_STRIDE_C = 1536
REDUCTION = BATCH * IN_T
BLOCK_R = triton.next_power_of_2(REDUCTION)
ZERO_BLOCK = triton.next_power_of_2(BATCH * (PAD_LEFT + PAD_RIGHT))


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
    active = r < 5808
    batch = r // 1452
    t = r - batch * 1452

    src_offsets = batch * (512 * 1452) + channel * 1452 + t
    mask_offsets = batch * 786432 + channel * 1536 + t
    padded_offsets = batch * (512 * 1493) + channel * 1493 + 20 + t

    src_vals = tl.load(src_ptr + src_offsets, mask=active, other=0.0)
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=active, other=0)
    scalar = tl.load(scalar_ptr)
    where_vals = tl.where(mask_vals, scalar, src_vals)

    tl.store(padded_ptr + padded_offsets, src_vals, mask=active)
    tl.store(where_ptr + src_offsets, where_vals, mask=active)

    z = tl.arange(0, ZERO_BLOCK_)
    z_active = z < 164
    z_batch = z // 41
    z_pos = z - z_batch * 41
    z_t = tl.where(z_pos < 20, z_pos, 1493 - 21 + (z_pos - 20))
    z_offsets = z_batch * (512 * 1493) + channel * 1493 + z_t
    zero = tl.full((ZERO_BLOCK_,), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(padded_ptr + z_offsets, zero, mask=z_active)

    total = tl.sum(tl.where(active, where_vals.to(tl.float32), 0.0), axis=0)
    tl.store(sum_ptr + channel, total)


# a190a59b: (T([4,512,1452], bf16), T([4,512,1452], b8, stride=(786432,1536,1)), T([], bf16), S([4,512,1493]))
@oracle_impl(hardware="B200", point="a190a59b", BLOCK_R=BLOCK_R, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0 = inputs

    device = arg0_1.device
    slice_scatter = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_0),
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
        arg0_1,
        arg1_1,
        arg2_1,
        slice_scatter,
        where,
        reduced,
        BLOCK_R_=BLOCK_R,
        ZERO_BLOCK_=ZERO_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return slice_scatter, where, reduced
