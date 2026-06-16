"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ALBERT bf16 masked-LM softmax-backward update, including the ignore-index target mask, the explicit bf16 rounding of the sparse target gradient, the explicit bf16 rounding before the natural exp, the returned dense bf16 `[4096,30000]` update, its transposed alias, and the bf16-rounded f32 vocabulary column sum. Inductor currently materializes the dense pointwise producer and rereads it through a separate column reduction, while also expanding the one-hot target mask as generic elementwise work; it cannot do this today because scheduler/codegen does not form one materialize-plus-split-row reduction plan that keeps a returned side output, a transpose alias, one-hot algebra, and bf16/f32 cast boundaries coordinated. The fix is COOPERATIVE_SPLIT_K: add a guarded softmax-backward materialization template that emits row-tiled column partials from the same producer traversal and finalizes the vocab sum once."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 4096
N = 30000


@triton.jit
def _materialize_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_max_ptr,
    row_logsum_ptr,
    residual_ptr,
    out_ptr,
    M_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_group = tl.program_id(0)
    col_group = tl.program_id(1)

    rows = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    active = (rows < M_SIZE) & (cols < N_SIZE)
    offsets = rows * N_SIZE + cols

    scale = tl.load(numerator_ptr).to(tl.float32) / tl.load(denominator_ptr).to(tl.float32)
    neg_scale = (-scale).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    label = tl.load(labels_ptr + rows, mask=rows < M_SIZE, other=-100)
    valid = label != -100
    in_range = valid & (label >= 0) & (label < N_SIZE)
    row_sum = tl.where(in_range, neg_scale, 0.0)
    target_grad = tl.where((cols == label) & in_range, row_sum, 0.0)

    logits = tl.load(logits_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    row_max = tl.load(row_max_ptr + rows, mask=rows < M_SIZE, other=0.0).to(tl.float32)
    row_logsum = tl.load(row_logsum_ptr + rows, mask=rows < M_SIZE, other=0.0).to(tl.float32)
    centered = (logits - row_max - row_logsum).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    probs = tl.exp(centered.to(tl.float32))
    grad = (target_grad - probs * row_sum).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    residual = tl.load(residual_ptr + offsets, mask=active, other=0.0)
    out_bf16 = (residual.to(tl.float32) + grad.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + offsets, out_bf16, mask=active)


@triton.jit
def _partial_sum_kernel(
    out_ptr,
    partial_ptr,
    M_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_group = tl.program_id(0)
    col_group = tl.program_id(1)
    rows = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    active = (rows < M_SIZE) & (cols < N_SIZE)
    values = tl.load(
        out_ptr + rows * N_SIZE + cols,
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    partial = tl.sum(tl.where(active, values, 0.0), axis=0)
    out_cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        partial_ptr + row_group * N_SIZE + out_cols,
        partial,
        mask=out_cols < N_SIZE,
    )


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    N_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_group = tl.program_id(0)
    groups = tl.arange(0, GROUP_BLOCK)[:, None]
    cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    active = (groups < NUM_GROUPS) & (cols < N_SIZE)

    values = tl.load(
        partial_ptr + groups * N_SIZE + cols,
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(active, values, 0.0), axis=0)
    out_cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + out_cols, rounded, mask=out_cols < N_SIZE)


# a93d5c9d: ALBERT masked-LM softmax backward, rows=4096 vocab=30000.
@oracle_impl(
    hardware="B200",
    point="a93d5c9d",
    ROW_BLOCK=2,
    BLOCK_N=8192,
    SUM_ROW_BLOCK=256,
    SUM_BLOCK_N=64,
    FINAL_BLOCK_N=64,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_N: int,
    SUM_ROW_BLOCK: int,
    SUM_BLOCK_N: int,
    FINAL_BLOCK_N: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, *_ = inputs
    device = arg3_1.device

    out = torch.empty_strided(
        (M, N),
        (N, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((N,), (1,), device=device, dtype=torch.float32)
    num_groups = triton.cdiv(M, SUM_ROW_BLOCK)
    partial = torch.empty_strided(
        (num_groups, N),
        (N, 1),
        device=device,
        dtype=torch.float32,
    )

    _materialize_kernel[(triton.cdiv(M, ROW_BLOCK), triton.cdiv(N, BLOCK_N))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        out,
        M_SIZE=M,
        N_SIZE=N,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    _partial_sum_kernel[(num_groups, triton.cdiv(N, SUM_BLOCK_N))](
        out,
        partial,
        M_SIZE=M,
        N_SIZE=N,
        ROW_BLOCK=SUM_ROW_BLOCK,
        BLOCK_N=SUM_BLOCK_N,
        num_warps=8,
    )
    group_block = 1 << (num_groups - 1).bit_length()
    _final_sum_kernel[(triton.cdiv(N, FINAL_BLOCK_N),)](
        partial,
        sum_out,
        N_SIZE=N,
        NUM_GROUPS=num_groups,
        GROUP_BLOCK=group_block,
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=8,
    )

    return out, out.permute(1, 0), sum_out
