"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Demucs bf16 relu/slice/add/mask scope in one flat Triton pointwise kernel, including `relu(arg0_1) + arg1_1[:, :, 1426:-1426]` and the boolean `relu(arg0_1) <= 0` side output while sharing the single `arg0_1` load. Inductor already lowers the full captured scope to a fused pointwise kernel that reads both inputs and stores both outputs; it cannot materially improve this local repro through scheduler fusion, scatter/reduce, split-K, algebraic elimination, or recomputation because runtime is dominated by the mandatory dense pointwise memory traffic. The fix is BANDWIDTH_BOUND: record this as a pointwise memory floor unless broader bf16 pointwise codegen or launch/allocation overhead changes move both paths."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 92844
M = 95696
SLICE_START = 1426
TOTAL = 4 * 64 * N


@triton.jit
def _relu_slice_add_mask_kernel(
    arg0_ptr,
    arg1_ptr,
    add_out_ptr,
    mask_out_ptr,
    N_: tl.constexpr,
    M_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    TOTAL_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < TOTAL_

    cols = offsets % N_
    rows = offsets // N_
    arg0 = tl.load(arg0_ptr + offsets, mask=valid, other=0.0)
    sliced = tl.load(arg1_ptr + rows * M_ + SLICE_START_ + cols, mask=valid, other=0.0)
    relu = tl.maximum(arg0, 0.0)

    tl.store(add_out_ptr + offsets, relu + sliced, mask=valid)
    tl.store(mask_out_ptr + offsets, arg0 <= 0.0, mask=valid)


# 3d3e6f4d: (T([4,64,92844], bf16), T([4,64,95696], bf16))
@oracle_impl(hardware="B200", point="3d3e6f4d", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    arg0_1, arg1_1 = inputs
    add_out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    mask_out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    _relu_slice_add_mask_kernel[(triton.cdiv(TOTAL, BLOCK),)](
        arg0_1,
        arg1_1,
        add_out,
        mask_out,
        N_=N,
        M_=M,
        SLICE_START_=SLICE_START,
        TOTAL_=TOTAL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return add_out, mask_out
