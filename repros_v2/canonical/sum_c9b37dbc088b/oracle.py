"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle stores the complete bf16 `where(arg2 <= 0, arg3, arg0 + arg1)` output and accumulates the same bf16-rounded values into per-channel split reductions in one channels-last pass, whereas Inductor materializes the returned `where` tensor separately from the channel sum for this multi-output graph; Inductor cannot do this today because its scheduler does not fuse a returned pointwise producer with a reduction consumer while preserving the materialized producer and choosing channels-last split-K partials; the fix is SCHEDULER_FUSION: add multi-output pointwise-plus-reduction fusion that emits the materialized store and cooperative per-channel partial reduction from the same traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _where_sum_partials_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    fill_ptr,
    where_ptr,
    partials_ptr,
    C: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    k_offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    offsets = k_offsets[:, None] * C + c_offsets[None, :]
    mask = (k_offsets[:, None] < K_TOTAL) & (c_offsets[None, :] < C)

    predicate = tl.load(arg2_ptr + offsets, mask=mask, other=1.0)
    take_fill = predicate <= 0.0
    lhs = tl.load(arg0_ptr + offsets, mask=mask & ~take_fill, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offsets, mask=mask & ~take_fill, other=0.0).to(tl.float32)
    add = (lhs + rhs).to(tl.bfloat16)
    fill = tl.load(fill_ptr)
    selected = tl.where(take_fill, fill, add)

    tl.store(where_ptr + offsets, selected, mask=mask)
    partial = tl.sum(tl.where(mask, selected.to(tl.float32), 0.0), axis=0)
    tl.store(
        partials_ptr + tl.program_id(0) * C + c_offsets,
        partial,
        mask=c_offsets < C,
    )


@triton.jit
def _finish_sum_kernel(
    partials_ptr,
    out_ptr,
    C: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial_offsets = tl.arange(0, BLOCK_PARTIALS)
    mask = (partial_offsets[:, None] < NUM_PARTIALS) & (c_offsets[None, :] < C)
    partials = tl.load(
        partials_ptr + partial_offsets[:, None] * C + c_offsets[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(partials, axis=0)
    tl.store(out_ptr + c_offsets, total, mask=c_offsets < C)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _launch(inputs, *, BLOCK_K, BLOCK_C, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    k_total = n * h * w
    num_partials = triton.cdiv(k_total, BLOCK_K)

    where_out = torch.empty_strided(
        (n, c, h, w),
        tuple(int(s) for s in arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty((num_partials, c), device=arg0_1.device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)

    grid = (num_partials, triton.cdiv(c, BLOCK_C))
    _where_sum_partials_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        where_out,
        partials,
        C=c,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _finish_sum_kernel[(triton.cdiv(c, BLOCK_C),)](
        partials,
        sum_out,
        C=c,
        NUM_PARTIALS=num_partials,
        BLOCK_PARTIALS=_next_power_of_2(num_partials),
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    return where_out, sum_out


# 398bc680: (T([32,16,55,55], bf16, stride=(48400,1,880,16)), T([32,16,55,55], bf16, stride=(48400,1,880,16)), T([32,16,55,55], bf16, stride=(48400,1,880,16)), T([], bf16))
@oracle_impl(hardware="B200", point="398bc680", BLOCK_K=256, BLOCK_C=16, num_warps=8)
# d20879a4: (T([32,32,27,27], bf16, stride=(23328,1,864,32)), T([32,32,27,27], bf16, stride=(23328,1,864,32)), T([32,32,27,27], bf16, stride=(23328,1,864,32)), T([], bf16))
@oracle_impl(hardware="B200", point="d20879a4", BLOCK_K=256, BLOCK_C=16, num_warps=8)
# a017535d: (T([32,48,13,13], bf16, stride=(8112,1,624,48)), T([32,48,13,13], bf16, stride=(8112,1,624,48)), T([32,48,13,13], bf16, stride=(8112,1,624,48)), T([], bf16))
@oracle_impl(hardware="B200", point="a017535d", BLOCK_K=128, BLOCK_C=16, num_warps=4)
# 6487a6cf: (T([32,64,13,13], bf16, stride=(10816,1,832,64)), T([32,64,13,13], bf16, stride=(10816,1,832,64)), T([32,64,13,13], bf16, stride=(10816,1,832,64)), T([], bf16))
@oracle_impl(hardware="B200", point="6487a6cf", BLOCK_K=128, BLOCK_C=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_K, BLOCK_C, num_warps):
    return _launch(inputs, BLOCK_K=BLOCK_K, BLOCK_C=BLOCK_C, num_warps=num_warps)
