"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete SqueezeNet bf16 max-pool-backward scatter feeding two returned masked channel slices and their f32 channel sums by reverse-gathering the structured scatter destinations directly from the captured low-memory pool offsets, whereas Inductor materializes the large f32 zero/scatter_add buffer, casts it to bf16, slices it, applies both masks, and then rereads both returned tensors for separate reductions. Inductor cannot do this today because scheduler/codegen treats `_low_memory_max_pool_offsets_to_indices` plus `scatter_add` as an opaque dense producer instead of a fixed-window max-pool scatter-reduce with live side-output and reduction consumers while preserving bf16 cast boundaries. The fix is SCATTER_REDUCE: add a guarded low-memory max-pool-backward lowering that recognizes constant 3x3/stride-2 scatter destinations and fuses the bf16 slice/mask materializations with the compatible channel reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 32
IN_C = 128
OUT_C = 64
SRC_H = 27
SRC_W = 27
DST_H = 55
DST_W = 55
SRC_HW = SRC_H * SRC_W
DST_HW = DST_H * DST_W
R = N * DST_HW
KERNEL = 3
STRIDE = 2


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _dual_pool_scatter_partials_kernel(
    src_ptr,
    offsets_ptr,
    mask_hi_ptr,
    fill_ptr,
    mask_lo_ptr,
    out_hi_ptr,
    out_lo_ptr,
    partial_ptr,
    R_SIZE: tl.constexpr,
    OUT_C_SIZE: tl.constexpr,
    SRC_H_SIZE: tl.constexpr,
    SRC_W_SIZE: tl.constexpr,
    DST_H_SIZE: tl.constexpr,
    DST_W_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STRIDE_SIZE: tl.constexpr,
    SRC_SN: tl.constexpr,
    SRC_SC: tl.constexpr,
    SRC_SH: tl.constexpr,
    SRC_SW: tl.constexpr,
    OFF_SN: tl.constexpr,
    OFF_SC: tl.constexpr,
    OFF_SH: tl.constexpr,
    OFF_SW: tl.constexpr,
    MASK_SN: tl.constexpr,
    MASK_SC: tl.constexpr,
    MASK_SH: tl.constexpr,
    MASK_SW: tl.constexpr,
    OUT_SN: tl.constexpr,
    OUT_SC: tl.constexpr,
    OUT_SH: tl.constexpr,
    OUT_SW: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    c_block = tl.program_id(1)
    row_base = tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    col_active = cols < OUT_C_SIZE
    fill = tl.load(fill_ptr)
    acc_hi = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_lo = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.range(0, GROUP_R, BLOCK_R):
        rows = group * GROUP_R + start + row_base
        row_active = rows < R_SIZE
        n = rows // (DST_H_SIZE * DST_W_SIZE)
        spatial = rows - n * (DST_H_SIZE * DST_W_SIZE)
        oh = spatial // DST_W_SIZE
        ow = spatial - oh * DST_W_SIZE
        active = row_active[:, None] & col_active[None, :]

        scatter_hi = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
        scatter_lo = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
        src_h0 = oh // STRIDE_SIZE
        src_w0 = ow // STRIDE_SIZE

        for dy in tl.static_range(0, 2):
            for dx in tl.static_range(0, 2):
                src_h = src_h0 - dy
                src_w = src_w0 - dx
                local_h = oh - src_h * STRIDE_SIZE
                local_w = ow - src_w * STRIDE_SIZE
                valid_row = (
                    row_active
                    & (src_h >= 0)
                    & (src_w >= 0)
                    & (local_h >= 0)
                    & (local_h < KERNEL_SIZE)
                    & (local_w >= 0)
                    & (local_w < KERNEL_SIZE)
                )
                valid_row = valid_row & (src_h < SRC_H_SIZE) & (src_w < SRC_W_SIZE)
                candidate = valid_row[:, None] & col_active[None, :]
                local_offset = local_h * KERNEL_SIZE + local_w

                lo_offsets = (
                    n[:, None] * OFF_SN
                    + cols[None, :] * OFF_SC
                    + src_h[:, None] * OFF_SH
                    + src_w[:, None] * OFF_SW
                )
                hi_cols = cols + OUT_C_SIZE
                hi_offsets = (
                    n[:, None] * OFF_SN
                    + hi_cols[None, :] * OFF_SC
                    + src_h[:, None] * OFF_SH
                    + src_w[:, None] * OFF_SW
                )
                lo_match = (
                    tl.load(offsets_ptr + lo_offsets, mask=candidate, other=-1).to(
                        tl.int32
                    )
                    == local_offset[:, None]
                )
                hi_match = (
                    tl.load(offsets_ptr + hi_offsets, mask=candidate, other=-1).to(
                        tl.int32
                    )
                    == local_offset[:, None]
                )

                src_lo_offsets = (
                    n[:, None] * SRC_SN
                    + cols[None, :] * SRC_SC
                    + src_h[:, None] * SRC_SH
                    + src_w[:, None] * SRC_SW
                )
                src_hi_offsets = (
                    n[:, None] * SRC_SN
                    + hi_cols[None, :] * SRC_SC
                    + src_h[:, None] * SRC_SH
                    + src_w[:, None] * SRC_SW
                )
                scatter_lo += tl.load(
                    src_ptr + src_lo_offsets,
                    mask=candidate & lo_match,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                scatter_hi += tl.load(
                    src_ptr + src_hi_offsets,
                    mask=candidate & hi_match,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)

        out_offsets = (
            n[:, None] * OUT_SN
            + cols[None, :] * OUT_SC
            + oh[:, None] * OUT_SH
            + ow[:, None] * OUT_SW
        )
        mask_offsets = (
            n[:, None] * MASK_SN
            + cols[None, :] * MASK_SC
            + oh[:, None] * MASK_SH
            + ow[:, None] * MASK_SW
        )
        take_hi = tl.load(mask_hi_ptr + mask_offsets, mask=active, other=0) != 0
        take_lo = tl.load(mask_lo_ptr + mask_offsets, mask=active, other=0) != 0
        hi_bf16 = scatter_hi.to(tl.bfloat16, fp_downcast_rounding="rtne")
        lo_bf16 = scatter_lo.to(tl.bfloat16, fp_downcast_rounding="rtne")
        out_hi = tl.where(take_hi, fill, hi_bf16)
        out_lo = tl.where(take_lo, fill, lo_bf16)

        tl.store(out_hi_ptr + out_offsets, out_hi, mask=active)
        tl.store(out_lo_ptr + out_offsets, out_lo, mask=active)
        acc_hi += tl.sum(tl.where(active, out_hi.to(tl.float32), 0.0), axis=0)
        acc_lo += tl.sum(tl.where(active, out_lo.to(tl.float32), 0.0), axis=0)

    partial_offsets = group * OUT_C_SIZE + cols
    tl.store(partial_ptr + partial_offsets, acc_hi, mask=col_active)
    tl.store(
        partial_ptr + NUM_GROUPS * OUT_C_SIZE + partial_offsets,
        acc_lo,
        mask=col_active,
    )


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_hi_ptr,
    sum_lo_ptr,
    NUM_GROUPS: tl.constexpr,
    OUT_C_SIZE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, BLOCK_GROUPS)
    active = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < OUT_C_SIZE)
    offsets = groups[:, None] * OUT_C_SIZE + cols[None, :]

    partial_hi = tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    partial_lo = tl.load(
        partial_ptr + NUM_GROUPS * OUT_C_SIZE + offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    sum_hi = tl.sum(partial_hi, axis=0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    sum_lo = tl.sum(partial_lo, axis=0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    tl.store(sum_hi_ptr + cols, sum_hi, mask=cols < OUT_C_SIZE)
    tl.store(sum_lo_ptr + cols, sum_lo, mask=cols < OUT_C_SIZE)


# (T([32,128,27,27], bf16, stride=(93312,1,3456,128)), T([32,128,27,27], i8, stride=(93312,1,3456,128), gen=Index(9)), T([32,64,55,55], b8, stride=(193600,1,3520,64)), T([], bf16), T([32,64,55,55], b8, stride=(193600,1,3520,64)))
@oracle_impl(
    hardware="B200",
    point="387bfba2",
    GROUP_R=256,
    BLOCK_R=16,
    BLOCK_C=16,
    FINAL_BLOCK_C=16,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    GROUP_R: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    device = arg0_1.device
    out_stride = tuple(int(s) for s in arg2_1.stride())
    out_hi = torch.empty_strided(
        (N, OUT_C, DST_H, DST_W),
        out_stride,
        device=device,
        dtype=torch.bfloat16,
    )
    out_lo = torch.empty_strided(
        (N, OUT_C, DST_H, DST_W),
        out_stride,
        device=device,
        dtype=torch.bfloat16,
    )
    sum_hi = torch.empty_strided((OUT_C,), (1,), device=device, dtype=torch.float32)
    sum_lo = torch.empty_strided((OUT_C,), (1,), device=device, dtype=torch.float32)
    num_groups = triton.cdiv(R, GROUP_R)
    partial = torch.empty(
        (2, num_groups, OUT_C),
        device=device,
        dtype=torch.float32,
    )

    src_stride = tuple(int(s) for s in arg0_1.stride())
    off_stride = tuple(int(s) for s in arg1_1.stride())
    mask_stride = tuple(int(s) for s in arg2_1.stride())

    _dual_pool_scatter_partials_kernel[(num_groups, triton.cdiv(OUT_C, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out_hi,
        out_lo,
        partial,
        R_SIZE=R,
        OUT_C_SIZE=OUT_C,
        SRC_H_SIZE=SRC_H,
        SRC_W_SIZE=SRC_W,
        DST_H_SIZE=DST_H,
        DST_W_SIZE=DST_W,
        KERNEL_SIZE=KERNEL,
        STRIDE_SIZE=STRIDE,
        SRC_SN=src_stride[0],
        SRC_SC=src_stride[1],
        SRC_SH=src_stride[2],
        SRC_SW=src_stride[3],
        OFF_SN=off_stride[0],
        OFF_SC=off_stride[1],
        OFF_SH=off_stride[2],
        OFF_SW=off_stride[3],
        MASK_SN=mask_stride[0],
        MASK_SC=mask_stride[1],
        MASK_SH=mask_stride[2],
        MASK_SW=mask_stride[3],
        OUT_SN=out_stride[0],
        OUT_SC=out_stride[1],
        OUT_SH=out_stride[2],
        OUT_SW=out_stride[3],
        NUM_GROUPS=num_groups,
        GROUP_R=GROUP_R,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(OUT_C, FINAL_BLOCK_C),)](
        partial,
        sum_hi,
        sum_lo,
        NUM_GROUPS=num_groups,
        OUT_C_SIZE=OUT_C,
        BLOCK_GROUPS=_next_power_of_2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )

    return out_hi, sum_hi, out_lo, sum_lo
