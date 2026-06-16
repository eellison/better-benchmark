"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle materializes the returned channels-last bf16 SiLU-backward tensor and accumulates per-channel partial sums from the same bf16-rounded values in each producer tile, whereas Inductor materializes the pointwise tensor and schedules the sibling channel reduction through generic reduction work over that producer; Inductor cannot do this today because its scheduler/codegen cannot coordinate a live returned bf16 side output with split-NHW channel-reduction partials while preserving the explicit sigmoid and bf16 cast boundary before the sum; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to emit a fused materialize-plus-partial-reduction producer for returned tensors with sibling reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _materialize_partial_sum_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    partial_ptr,
    sum_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    GROUP_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STORE_DIRECT_SUM: tl.constexpr,
):
    group = tl.program_id(0)
    c_block = tl.program_id(1)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.range(0, GROUP_R, BLOCK_R):
        rows = group * GROUP_R + start + tl.arange(0, BLOCK_R)
        offsets = rows[:, None] * C + cols[None, :]
        mask = (rows[:, None] < R) & col_mask[None, :]

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(x)
        value = grad * sig * (x * (1.0 - sig) + 1.0)
        value_bf16 = value.to(tl.bfloat16)

        tl.store(out_ptr + offsets, value_bf16, mask=mask)
        acc += tl.sum(tl.where(mask, value_bf16.to(tl.float32), 0.0), axis=0)

    rounded = acc.to(tl.bfloat16).to(tl.float32)
    if STORE_DIRECT_SUM:
        tl.store(sum_ptr + cols, rounded, mask=col_mask)
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
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < C)


# (T([128,16,112,112], bf16, stride=(200704,1,1792,16)), ...)
@oracle_impl(hardware="B200", point="c5cf0dd3", GROUP_R=1024, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=4, num_warps=8)
# (T([128,32,112,112], bf16, stride=(401408,1,3584,32)), ...)
@oracle_impl(hardware="B200", point="f40368ca", GROUP_R=1024, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=4, num_warps=8)
# (T([128,64,112,112], bf16, stride=(802816,1,7168,64)), ...)
@oracle_impl(hardware="B200", point="934b0f73", GROUP_R=1024, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=4, num_warps=8)
# (T([128,64,56,56], bf16, stride=(200704,1,3584,64)), ...)
@oracle_impl(hardware="B200", point="96449bbf", GROUP_R=1024, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=8, num_warps=8)
# (T([128,128,56,56], bf16, stride=(401408,1,7168,128)), ...)
@oracle_impl(hardware="B200", point="d34f6e69", GROUP_R=1024, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=8, num_warps=8)
# (T([128,128,28,28], bf16, stride=(100352,1,3584,128)), ...)
@oracle_impl(hardware="B200", point="13a2d815", GROUP_R=1024, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,384,28,28], bf16, stride=(301056,1,10752,384)), ...)
@oracle_impl(hardware="B200", point="f961de61", GROUP_R=1024, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,384,14,14], bf16, stride=(75264,1,5376,384)), ...)
@oracle_impl(hardware="B200", point="7c24d0c8", GROUP_R=512, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,384,7,7], bf16, stride=(18816,1,2688,384)), ...)
@oracle_impl(hardware="B200", point="5c7adf25", GROUP_R=256, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,8,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="03f3f1e3", GROUP_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,4,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="9926e6d2", GROUP_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,6,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="5045fc42", GROUP_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,10,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="6ced833e", GROUP_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,20,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="cadd9933", GROUP_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,28,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="00335e2b", GROUP_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,48,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="78ac4aa7", GROUP_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
def oracle_forward(
    inputs,
    *,
    GROUP_R: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    grad, x = inputs
    n = int(grad.shape[0])
    c = int(grad.shape[1])
    h = int(grad.shape[2])
    w = int(grad.shape[3])
    r = n * h * w

    out = torch.empty_strided(
        tuple(grad.shape),
        tuple(grad.stride()),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((c,), device=grad.device, dtype=torch.float32)
    num_groups = triton.cdiv(r, GROUP_R)
    direct_sum = num_groups == 1
    partial = sum_out if direct_sum else torch.empty((num_groups, c), device=grad.device, dtype=torch.float32)

    _materialize_partial_sum_kernel[(num_groups, triton.cdiv(c, BLOCK_C))](
        grad,
        x,
        out,
        partial,
        sum_out,
        R=r,
        C=c,
        GROUP_R=GROUP_R,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        STORE_DIRECT_SUM=direct_sum,
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

    return out, sum_out
