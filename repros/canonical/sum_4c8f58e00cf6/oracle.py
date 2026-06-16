"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 masked materialization plus channel-sum scope in one Triton column-reduction kernel, including the returned bf16 scalar zero, the visible `[512,1280,1,1]` `where(mask, 0, view(arg0))` tensor, and the returned f32 channel sum over those bf16 producer values, whereas Inductor lowers the full/view/where/sum/cast graph through generic materialization and reduction schedules; Inductor cannot do this today because its scheduler does not form one full-scope plan that shares a returned bf16 masked producer with its compatible channel reduction while preserving the scalar side output and compiled f32 reduction envelope; the fix is SCHEDULER_FUSION: teach reduction scheduling to keep such masked producers virtual, emit required visible outputs, and finalize the dependent reduction from the same traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _masked_materialize_sum_kernel(
    x_ptr,
    mask_ptr,
    full_ptr,
    where_ptr,
    sum_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, M)
    active = cols < N
    offsets = rows[:, None] * N + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=active[None, :], other=0.0)
    mask_values = tl.load(mask_ptr + offsets, mask=active[None, :], other=1) != 0
    selected = tl.where(mask_values, 0.0, x).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(where_ptr + offsets, selected, mask=active[None, :])
    reduced = tl.sum(selected.to(tl.float32), axis=0)
    tl.store(sum_ptr + cols, reduced, mask=active)
    tl.store(full_ptr, 0.0, mask=tl.program_id(0) == 0)


@oracle_impl(hardware="B200", point="86b9700f", BLOCK_N=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0 = inputs
    where_shape = tuple(int(dim) for dim in _shape_param_0)
    m = int(where_shape[0])
    n = int(where_shape[1])

    full = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.bfloat16)
    where = torch.empty_strided(
        where_shape,
        (n, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((n,), (1,), device=arg0_1.device, dtype=torch.float32)

    _masked_materialize_sum_kernel[(triton.cdiv(n, BLOCK_N),)](
        arg0_1,
        arg1_1,
        full,
        where,
        sum_out,
        M=m,
        N=n,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return full, where, sum_out
