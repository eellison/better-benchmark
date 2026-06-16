"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Demucs bf16 gated cat-and-sum scope by fusing Inductor's promoted bf16-input materialization path, fp32 sigmoid gate, two branch products, returned bf16 `[4, 1024, 1493]` cat tensor, and sibling `[1024]` bf16-rounded reduction from the same producer tiles, whereas Inductor lowers the slice/sigmoid/sub/mul/cat/cast/sum graph as generic pointwise materialization followed by a separate reduction over the bf16 cat output; Inductor cannot do this today because its scheduler does not form one multi-output plan that shares the gate producer across disjoint cat slices while preserving the required bf16 output and bf16 sum-result rounding boundaries; the fix is SCHEDULER_FUSION: teach reduction scheduling to sink static channel-cat producers into a store-and-partial-reduce template that writes the observable bf16 tensor and finalizes the rounded channel sums directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 512
C_TOTAL = 1024
TIME = 1493
K_TOTAL = BATCH * TIME


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def _direct_channel_kernel(
    lhs_ptr,
    rhs_ptr,
    gate_ptr,
    out_ptr,
    sum_out_ptr,
    K_TOTAL_: tl.constexpr,
    C_HALF_: tl.constexpr,
    C_TOTAL_: tl.constexpr,
    TIME_: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    c = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_T)
    acc_first = tl.full((), 0.0, tl.float32)
    acc_second = tl.full((), 0.0, tl.float32)

    for start in tl.range(0, K_TOTAL_, BLOCK_T):
        k = start + lanes
        batch = k // TIME_
        time = k - batch * TIME_
        active = k < K_TOTAL_

        half_offsets = batch * (C_HALF_ * TIME_) + c * TIME_ + time
        full_offsets = batch * (C_TOTAL_ * TIME_) + c * TIME_ + time

        lhs = tl.load(lhs_ptr + half_offsets, mask=active, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + half_offsets, mask=active, other=0.0).to(tl.float32)
        add_unrounded = _add_rn_f32(lhs, rhs)
        add_rounded = add_unrounded.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

        gate_a = tl.load(gate_ptr + full_offsets, mask=active, other=0.0).to(tl.float32)
        gate_b = tl.load(
            gate_ptr + full_offsets + C_HALF_ * TIME_,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        sigmoid = _div_rn_f32(1.0, _add_rn_f32(1.0, libdevice.exp(_sub_rn_f32(0.0, gate_b))))
        gate_deriv = _mul_rn_f32(_sub_rn_f32(1.0, sigmoid), sigmoid)

        first = _mul_rn_f32(sigmoid, add_unrounded).to(tl.bfloat16, fp_downcast_rounding="rtne")
        second = _mul_rn_f32(
            _mul_rn_f32(gate_deriv, gate_a),
            add_unrounded,
        ).to(tl.bfloat16, fp_downcast_rounding="rtne")
        first_for_sum = _mul_rn_f32(sigmoid, add_rounded).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        )
        second_for_sum = _mul_rn_f32(
            _mul_rn_f32(gate_deriv, gate_a),
            add_rounded,
        ).to(tl.bfloat16, fp_downcast_rounding="rtne")

        tl.store(out_ptr + full_offsets, first, mask=active)
        tl.store(out_ptr + full_offsets + C_HALF_ * TIME_, second, mask=active)
        acc_first += tl.sum(tl.where(active, first_for_sum.to(tl.float32), 0.0), axis=0)
        acc_second += tl.sum(tl.where(active, second_for_sum.to(tl.float32), 0.0), axis=0)

    tl.store(sum_out_ptr + c, acc_first.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32))
    tl.store(
        sum_out_ptr + C_HALF_ + c,
        acc_second.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
    )


@oracle_impl(
    hardware="B200",
    point="bdbb44ef",
    BLOCK_T=1024,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_T: int,
    num_warps: int,
):
    lhs, rhs, gate = inputs

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=gate.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((C_TOTAL,), (1,), device=gate.device, dtype=torch.float32)

    _direct_channel_kernel[(C_HALF,)](
        lhs,
        rhs,
        gate,
        out,
        sum_out,
        K_TOTAL_=K_TOTAL,
        C_HALF_=C_HALF,
        C_TOTAL_=C_TOTAL,
        TIME_=TIME,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, sum_out
