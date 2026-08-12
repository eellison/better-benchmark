"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 hard-swish materialization plus channel-sum scope in one Triton kernel, including the returned f32 scalar zero, the view-only `[512,1280] -> [512,1280,1,1]` input interpretation, f32 thresholded hard-swish formulation, explicit bf16 activation rounding, the returned contiguous bf16 activation, and the returned `[1280]` f32 channel sum over the bf16-rounded activation. Inductor currently lowers the hard-swish pointwise producer, visible bf16 activation store, and sibling `sum([0,2,3])` consumer through generic materialization and reduction scheduling; it cannot do this today because the scheduler does not keep a required singleton-spatial materialized side output and the compatible channel reduction in one full-scope plan while preserving the bf16 producer boundary. The fix is SCHEDULER_FUSION: recognize singleton-spatial hard-swish producers with visible materialization plus channel-sum consumers and emit a fused column-reduction/materialization kernel."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 512
COLS = 1280
OUT_SHAPE = (ROWS, COLS, 1, 1)
OUT_STRIDE = (COLS, 1, 1, 1)


@triton.jit
def _hardswish_materialize_sum_kernel(
    x_ptr,
    gate_ptr,
    zero_out_ptr,
    activation_ptr,
    sum_ptr,
    ROWS_N: tl.constexpr,
    COLS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_block = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < ROWS_N) & (cols[None, :] < COLS_N)
    offsets = rows[:, None] * COLS_N + cols[None, :]

    tl.store(zero_out_ptr, 0.0, mask=col_block == 0)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    linear_gate = gate / 3.0 + 0.5
    middle = x * linear_gate
    value = tl.where(gate < 3.0, middle, x)
    value = tl.where(gate <= -3.0, 0.0, value)
    activation = value.to(tl.bfloat16)

    tl.store(activation_ptr + offsets, activation, mask=mask)
    reduced = tl.sum(tl.where(mask, activation.to(tl.float32), 0.0), axis=0)
    tl.store(sum_ptr + cols, reduced, mask=cols < COLS_N)


@oracle_impl(hardware="B200", point="cf51a5e7", BLOCK_M=512, BLOCK_N=4, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    x, gate, shape0 = inputs
    del shape0

    zero = torch.empty_strided((), (), device=x.device, dtype=torch.float32)
    activation = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=torch.bfloat16,
    )
    sums = torch.empty_strided((COLS,), (1,), device=x.device, dtype=torch.float32)
    _hardswish_materialize_sum_kernel[(triton.cdiv(COLS, BLOCK_N),)](
        x,
        gate,
        zero,
        activation,
        sums,
        ROWS_N=ROWS,
        COLS_N=COLS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return zero, activation, sums
