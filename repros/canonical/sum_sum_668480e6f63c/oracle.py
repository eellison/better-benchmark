"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete XLNet bf16 permute/add/clone scope and the two sibling f32 `[16,64]` sums with one producer that writes the final `[8192,1024]` clone layout while emitting matching split-K partial sums for both inputs, followed by one co-finalizer for the two reductions, whereas Inductor emits separate two-stage reductions for each input and a separate pointwise clone/add kernel that rereads both tensors; Inductor cannot do this today because its reduction scheduler cannot coordinate compatible sibling reductions with a live layout-changing side output from the same permuted producer while preserving the bf16 add/cast boundary; the fix is COOPERATIVE_SPLIT_K: add a multi-output split-K producer/finalizer plan that shares permuted bf16 loads across reductions and required clone materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _clone_and_partial_sums_kernel(
    arg0_ptr,
    arg1_ptr,
    out_ptr,
    partial0_ptr,
    partial1_ptr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_group = tl.program_id(0)
    col_group = tl.program_id(1)

    rows = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    seq = rows // 16
    inner = rows - seq * 16
    head = cols // 64
    dim = cols - head * 64
    source_offsets = dim + 64 * seq + 32768 * head + 524288 * inner

    x0 = tl.load(arg0_ptr + source_offsets).to(tl.float32)
    x1 = tl.load(arg1_ptr + source_offsets).to(tl.float32)
    tl.store(out_ptr + rows * 1024 + cols, (x0 + x1).to(tl.bfloat16))

    out_cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)
    partial_offsets = row_group * 1024 + out_cols
    tl.store(partial0_ptr + partial_offsets, tl.sum(x0, axis=0))
    tl.store(partial1_ptr + partial_offsets, tl.sum(x1, axis=0))


@triton.jit
def _finalize_sums_kernel(
    partial0_ptr,
    partial1_ptr,
    sum0_ptr,
    sum1_ptr,
    FINAL_BLOCK_N: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * FINAL_BLOCK_N + tl.arange(0, FINAL_BLOCK_N)[None, :]
    groups = tl.arange(0, GROUP_BLOCK)[:, None]
    offsets = groups * 1024 + cols

    vals0 = tl.load(partial0_ptr + offsets).to(tl.float32)
    vals1 = tl.load(partial1_ptr + offsets).to(tl.float32)
    out_cols = tl.program_id(0) * FINAL_BLOCK_N + tl.arange(0, FINAL_BLOCK_N)
    tl.store(sum0_ptr + out_cols, tl.sum(vals0, axis=0))
    tl.store(sum1_ptr + out_cols, tl.sum(vals1, axis=0))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="28a9d256",
    ROW_BLOCK=128,
    BLOCK_N=64,
    FINAL_BLOCK_N=16,
    num_warps=4,
)
def oracle_forward(inputs, *, ROW_BLOCK, BLOCK_N, FINAL_BLOCK_N, num_warps):
    (
        arg0_1,
        arg1_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
    ) = inputs

    sum0 = torch.empty_strided(
        _shape_tuple(_shape_param_1),
        (64, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum1 = torch.empty_strided(
        _shape_tuple(_shape_param_3),
        (64, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out_shape = _shape_tuple(_shape_param_5)[1:]
    out = torch.empty_strided(
        out_shape,
        (1024, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    num_row_groups = triton.cdiv(8192, ROW_BLOCK)
    partial0 = torch.empty_strided(
        (num_row_groups, 1024),
        (1024, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    partial1 = torch.empty_strided(
        (num_row_groups, 1024),
        (1024, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _clone_and_partial_sums_kernel[(num_row_groups, triton.cdiv(1024, BLOCK_N))](
        arg0_1,
        arg1_1,
        out,
        partial0,
        partial1,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    _finalize_sums_kernel[(triton.cdiv(1024, FINAL_BLOCK_N),)](
        partial0,
        partial1,
        sum0,
        sum1,
        FINAL_BLOCK_N=FINAL_BLOCK_N,
        GROUP_BLOCK=num_row_groups,
        num_warps=4,
    )
    return sum0, sum1, out
