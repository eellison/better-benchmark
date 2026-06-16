"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 mask-scale hard-swish producer plus column-sum scope in one Triton column-reduction kernel, including the returned f32 scalar zero, the visible contiguous bf16 `[32,1280]` producer in Inductor's fused fp32-multiply envelope, its metadata-only `[1280,32]` permute alias, and the returned f32 vector after the captured reduction-side bf16-to-f32 round trip, whereas Inductor lowers the bool-scale, hard-swish-like gate, bf16 materialization, permute view, and sibling reduction through generic pointwise/reduction scheduling; Inductor cannot do this today because its scheduler does not fuse a shared pointwise producer into both a returned materialized tensor and a dependent reduction while preserving the compiled cast boundaries and alias output; the fix is SCHEDULER_FUSION: teach multi-output reduction scheduling to emit the producer, alias, and reduced epilogue from one shape-specialized plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _hardswish_materialize_sum_kernel(
    mask_ptr,
    scale_src_ptr,
    gate_ptr,
    full_ptr,
    out_ptr,
    sum_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, M)
    active = cols < N
    offsets = rows[:, None] * N + cols[None, :]

    mask = tl.load(mask_ptr + offsets, mask=active[None, :], other=0)
    scale_src = tl.load(scale_src_ptr + offsets, mask=active[None, :], other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=active[None, :], other=0.0).to(tl.float32)

    scaled_f32 = tl.where(mask != 0, scale_src * 1.25, 0.0)
    scaled_eager = scaled_f32.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    factor = gate / 3.0 + 0.5
    gated = scaled_f32 * factor
    value = tl.where(gate < 3.0, gated, scaled_f32)
    value = tl.where(gate <= -3.0, 0.0, value)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    gated_eager = scaled_eager * factor
    value_eager = tl.where(gate < 3.0, gated_eager, scaled_eager)
    value_eager = tl.where(gate <= -3.0, 0.0, value_eager)
    value_sum_bf16 = value_eager.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets, value_bf16, mask=active[None, :])
    reduced = tl.sum(value_sum_bf16.to(tl.float32), axis=0)
    tl.store(sum_ptr + cols, _round_to_bf16_f32(reduced), mask=active)
    tl.store(full_ptr, 0.0, mask=tl.program_id(0) == 0)


@oracle_impl(hardware="B200", point="c7ec772c", BLOCK_N=8, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0 = inputs
    n = int(_shape_param_0[0])
    m = int(arg1_1.shape[0])

    full = torch.empty_strided((), (), device=arg1_1.device, dtype=torch.float32)
    out = torch.empty_strided((m, n), (n, 1), device=arg1_1.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((n,), (1,), device=arg1_1.device, dtype=torch.float32)

    _hardswish_materialize_sum_kernel[(triton.cdiv(n, BLOCK_N),)](
        arg0_1,
        arg1_1,
        arg2_1,
        full,
        out,
        sum_out,
        M=m,
        N=n,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return full, out, out.permute(1, 0), sum_out
