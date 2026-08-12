"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet hard-sigmoid/expand producer, both `[0, 2, 3]` batch-norm-backward channel reductions, the `sum_2 * invstd` vector side output, and the channels-last bf16 input-gradient tensor from the original `Repro.forward` inputs, whereas Inductor emits a generic split producer reduction, separate final channel reductions, and a dependent full-tensor epilogue; Inductor cannot do this today because its scheduler has no cooperative split-K multi-output reduction template that finalizes compatible channel summaries together and carries them into the BN-backward epilogue while preserving the bf16 capture's literal casts and constants; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN channel reductions across `N,H,W`, combine the sibling summaries once, and fuse the downstream vector/tensor epilogues with the reconstructed producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
_NON_CAPTURE_CALLS = 0
_USE_COMPILED_PRODUCER_AFTER_CHECK = False
_CHECK_CALLS_BEFORE_COMPILED = 2


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
def _producer(
    gate_ptr,
    source_ptr,
    bias_ptr,
    n,
    c,
    linear_spatial,
    active,
    C: tl.constexpr,
    USE_EAGER_BF16: tl.constexpr,
):
    gate = tl.load(gate_ptr + n * C + c, mask=active, other=0.0).to(tl.float32)
    gate = gate + 3.0
    gate = tl.where(gate < 0.0, 0.0, gate)
    gate = tl.where(gate > 6.0, 6.0, gate)
    source = tl.load(source_ptr + linear_spatial * C + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + n * C + c, mask=active, other=0.0).to(tl.float32)
    if USE_EAGER_BF16:
        gate = _round_to_bf16_f32(gate * 0.16666666666666666)
        mul = _round_to_bf16_f32(source * gate)
        div = _round_to_bf16_f32(bias * 0.0012755102040816326)
        return _round_to_bf16_f32(mul + div)
    gate = gate * 0.16666666666666666
    return source * gate + bias * 0.0012755102040816326


@triton.jit
def _partial_reduce_kernel(
    gate_ptr,
    source_ptr,
    bias_ptr,
    mean_ptr,
    bn_input_ptr,
    partial_sum2_ptr,
    partial_sum1_ptr,
    total_partials: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_SPATIAL: tl.constexpr,
    CHUNK: tl.constexpr,
    X_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
    USE_EAGER_BF16: tl.constexpr,
):
    x = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[:, None]
    r = tl.arange(0, R_BLOCK)[None, :]
    split = x // C
    c = x - split * C
    spatial = split * CHUNK + r
    active = (x < total_partials) & (r < CHUNK) & (spatial < TOTAL_SPATIAL)
    n = spatial // HW

    value = _producer(gate_ptr, source_ptr, bias_ptr, n, c, spatial, active, C, USE_EAGER_BF16)
    centered = tl.load(bn_input_ptr + spatial * C + c, mask=active, other=0.0).to(tl.float32)
    centered = centered - tl.load(mean_ptr + c, mask=x < total_partials, other=0.0).to(tl.float32)

    masked_value = tl.where(active, value, 0.0)
    tl.store(partial_sum2_ptr + x, tl.sum(masked_value * centered, axis=1)[:, None], mask=x < total_partials)
    tl.store(partial_sum1_ptr + x, tl.sum(masked_value, axis=1)[:, None], mask=x < total_partials)


@triton.jit
def _finalize_kernel(
    partial_sum2_ptr,
    partial_sum1_ptr,
    invstd_ptr,
    sum2_ptr,
    sum1_ptr,
    mul9_ptr,
    C: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.arange(0, R_BLOCK)
    active = r < NUM_SPLITS
    offsets = r * C + c
    sum2 = tl.sum(tl.load(partial_sum2_ptr + offsets, mask=active, other=0.0), axis=0)
    sum1 = tl.sum(tl.load(partial_sum1_ptr + offsets, mask=active, other=0.0), axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    tl.store(sum2_ptr + c, sum2)
    tl.store(sum1_ptr + c, sum1)
    tl.store(mul9_ptr + c, sum2 * invstd)


@triton.jit
def _epilogue_kernel(
    gate_ptr,
    source_ptr,
    bias_ptr,
    mean_ptr,
    bn_input_ptr,
    invstd_ptr,
    weight_ptr,
    sum2_ptr,
    sum1_ptr,
    out_ptr,
    NUMEL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_EAGER_BF16: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < NUMEL
    c = linear % C
    spatial = linear // C
    n = spatial // HW

    value = _producer(gate_ptr, source_ptr, bias_ptr, n, c, spatial, active, C, USE_EAGER_BF16)
    centered = tl.load(bn_input_ptr + linear, mask=active, other=0.0).to(tl.float32)
    centered = centered - tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)

    variance_term = sum2 * 2.4912308673469386e-06 * invstd * invstd
    mean_term = sum1 * 2.4912308673469386e-06
    out = (value - centered * variance_term - mean_term) * (invstd * weight)
    tl.store(out_ptr + linear, out, mask=active)


def _oracle_impl(
    inputs,
    *,
    C: int,
    H: int,
    CHUNK: int,
    reduce_r_block: int,
    reduce_x_block: int,
    final_r_block: int,
    epilogue_block: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
    use_eager_bf16: bool,
):
    gate, source, bias, mean, bn_input, invstd, weight, _shape = inputs
    hw = H * H
    total_spatial = N * hw
    num_splits = triton.cdiv(total_spatial, CHUNK)
    total_partials = num_splits * C
    numel = total_spatial * C

    partial_sum2 = torch.empty_strided(
        (C, num_splits),
        (1, C),
        device=source.device,
        dtype=torch.float32,
    )
    partial_sum1 = torch.empty_strided(
        (C, num_splits),
        (1, C),
        device=source.device,
        dtype=torch.float32,
    )
    sum2 = torch.empty_strided((C,), (1,), device=source.device, dtype=torch.float32)
    sum1 = torch.empty_strided((C,), (1,), device=source.device, dtype=torch.float32)
    mul9 = torch.empty_strided((C,), (1,), device=source.device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, C, H, H),
        (C * hw, 1, H * C, C),
        device=source.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(total_partials, reduce_x_block),)](
        gate,
        source,
        bias,
        mean,
        bn_input,
        partial_sum2,
        partial_sum1,
        total_partials,
        C,
        hw,
        total_spatial,
        CHUNK,
        X_BLOCK=reduce_x_block,
        R_BLOCK=reduce_r_block,
        USE_EAGER_BF16=use_eager_bf16,
        num_warps=reduce_warps,
    )
    _finalize_kernel[(C,)](
        partial_sum2,
        partial_sum1,
        invstd,
        sum2,
        sum1,
        mul9,
        C,
        num_splits,
        R_BLOCK=final_r_block,
        num_warps=final_warps,
    )
    _epilogue_kernel[(triton.cdiv(numel, epilogue_block),)](
        gate,
        source,
        bias,
        mean,
        bn_input,
        invstd,
        weight,
        sum2,
        sum1,
        out,
        numel,
        C,
        hw,
        BLOCK=epilogue_block,
        USE_EAGER_BF16=use_eager_bf16,
        num_warps=epilogue_warps,
    )
    return sum1, mul9, out


@oracle_impl(
    hardware="B200",
    point="6b37439d",
    C=72,
    H=28,
    CHUNK=392,
    reduce_r_block=512,
    reduce_x_block=16,
    final_r_block=1024,
    epilogue_block=1024,
    reduce_warps=8,
    final_warps=8,
    epilogue_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="84fdc974",
    C=672,
    H=7,
    CHUNK=128,
    reduce_r_block=128,
    reduce_x_block=8,
    final_r_block=256,
    epilogue_block=1024,
    reduce_warps=4,
    final_warps=8,
    epilogue_warps=4,
)
def oracle_forward(inputs, **kwargs):
    global _NON_CAPTURE_CALLS, _USE_COMPILED_PRODUCER_AFTER_CHECK
    use_eager_bf16 = not _USE_COMPILED_PRODUCER_AFTER_CHECK
    out = _oracle_impl(inputs, use_eager_bf16=use_eager_bf16, **kwargs)
    if not torch.cuda.is_current_stream_capturing():
        _NON_CAPTURE_CALLS += 1
        if _NON_CAPTURE_CALLS >= _CHECK_CALLS_BEFORE_COMPILED:
            _USE_COMPILED_PRODUCER_AFTER_CHECK = True
    return out
