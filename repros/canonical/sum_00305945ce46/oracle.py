"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete AlexNet bf16 ReLU-backward mask producer once, including the observable bf16 scalar zero, the `arg0 <= 0` select into returned contiguous `[128,4096]` storage, the metadata-only `[4096,128]` transpose alias, and the sibling dim-0 f32 column sum over the bf16 selected values followed by the captured bf16-to-f32 round trip. Inductor lowers the shared `where(arg0 <= 0, 0, arg1)` producer, transpose view, and column reduction through generic multi-output reduction scheduling with avoidable producer replay/materialization overhead; it cannot do this today because its scheduler does not form an alias-aware full-scope producer-store plus reduction plan for a value that is both returned and reduced while preserving explicit bf16 output and sum-rounding boundaries. The fix is SCHEDULER_FUSION: teach multi-output reduction scheduling to keep the masked bf16 producer virtual, emit the visible base storage once, return view aliases, and finalize compatible column reductions from the same traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 128
N = 4096


@triton.jit
def _relu_mask_store_sum_kernel(
    mask_input_ptr,
    source_ptr,
    full_ptr,
    out_ptr,
    sum_ptr,
    M_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    active = (rows[:, None] < M_SIZE) & (cols[None, :] < N_SIZE)
    offsets = rows[:, None] * N_SIZE + cols[None, :]

    mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0)
    zero = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    values = tl.where(mask_input <= 0.0, zero, source)

    tl.store(out_ptr + offsets, values, mask=active)
    total = tl.sum(tl.where(active, values.to(tl.float32), 0.0), axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < N_SIZE)
    zero_scalar = tl.full((), 0.0, tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(full_ptr, zero_scalar, mask=tl.program_id(0) == 0)


@oracle_impl(hardware="B200", point="fb7c5a2a", BLOCK_M=128, BLOCK_N=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    mask_input, source, shape = inputs
    del shape

    full = torch.empty_strided((), (), device=source.device, dtype=torch.bfloat16)
    out = torch.empty_strided((M, N), (N, 1), device=source.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((N,), (1,), device=source.device, dtype=torch.float32)

    _relu_mask_store_sum_kernel[(triton.cdiv(N, BLOCK_N),)](
        mask_input,
        source,
        full,
        out,
        sum_out,
        M_SIZE=M,
        N_SIZE=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return full, out, torch.as_strided(out, (N, M), (1, N)), sum_out
