"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GPT-Neo bf16 masked row-reduction scope in one Triton kernel, including the `[512,128,128] -> [32,16,128,128]` metadata view, bf16-to-f32 activation cast, f32 product with the attention tensor, last-dimension f32 sum, exact `prims.fma(-arg1, sum, product)` placement, final bf16 cast, sliced `[1,1,2048,2048] -> [1,1,128,128]` bool mask broadcast, returned bf16 scalar zero, and returned contiguous `[512,128,128]` bf16 output view, whereas Inductor already emits a similar fused row-reduction/pointwise kernel for this local graph; Inductor cannot materially improve this repro through scheduler fusion alone because the mandatory input reads, 128-wide row reductions, mask load, bf16 output store, and launch overhead dominate after the metadata-only views are erased; the fix is BANDWIDTH_BOUND: record this as an at-floor fused row-reduction case unless broader reduction codegen or memory-traffic improvements move both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 16
SEQ = 128
TOTAL_ROWS = BATCH * HEADS * SEQ
MASK_STRIDE = 2048
VIEW_SHAPE = (BATCH, HEADS, SEQ, SEQ)
VIEW_STRIDE = (HEADS * SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@triton.jit
def _row_fma_mask_bf16_kernel(
    x_ptr,
    weight_ptr,
    mask_ptr,
    full_ptr,
    out_ptr,
    seq_size: tl.constexpr,
    mask_stride: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols = tl.arange(0, seq_size)
    offsets = rows[:, None] * seq_size + cols[None, :]

    x = tl.load(x_ptr + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    product = x * weight
    row_sum = tl.sum(product, axis=1)[:, None].to(tl.float32)

    masked_q = rows % seq_size
    keep = tl.load(mask_ptr + masked_q[:, None] * mask_stride + cols[None, :]).to(tl.int1)
    value = tl.fma(-weight, row_sum, product).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    zero = tl.full((ROWS_PER_PROGRAM, seq_size), 0.0, tl.float32).to(tl.bfloat16)
    selected = tl.where(keep, value, zero)

    tl.store(out_ptr + offsets, selected)
    tl.store(full_ptr, tl.full((), 0.0, tl.float32).to(tl.bfloat16), mask=tl.program_id(0) == 0)


# 4e163e19: (T([512,128,128], bf16), T([32,16,128,128], f32), T([1,1,2048,2048], b8), ...)
@oracle_impl(hardware="B200", point="4e163e19", ROWS_PER_PROGRAM=32, num_warps=4)
def oracle_forward(inputs, *, ROWS_PER_PROGRAM: int, num_warps: int):
    x, weight, mask, _shape_param_0, out_shape = inputs
    full = torch.empty_strided((), (), device=x.device, dtype=torch.bfloat16)
    out_base = torch.empty_strided(
        VIEW_SHAPE,
        VIEW_STRIDE,
        device=x.device,
        dtype=torch.bfloat16,
    )

    _row_fma_mask_bf16_kernel[(triton.cdiv(TOTAL_ROWS, ROWS_PER_PROGRAM),)](
        x,
        weight,
        mask,
        full,
        out_base,
        seq_size=SEQ,
        mask_stride=MASK_STRIDE,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        num_warps=num_warps,
        num_stages=3,
    )
    return full, out_base.view(tuple(int(dim) for dim in out_shape))
