"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Repro.forward tuple by fusing the avg-pool-backward/add producer into a structured max-pool scatter, handling the ReLU-masked full-fill path analytically in the reductions and epilogue, and overwriting the channels-last scatter buffer with the dependent BN-backward output, whereas Inductor currently materializes the [24576, 5041] scatter_add result and schedules the fill/where, two channel reductions, and dependent epilogue as separate tensor work; Inductor cannot do this today because the scheduler/codegen has no structured max-pool-backward scatter_reduce template that preserves the low-memory pool-offset mapping while fusing sibling channel reductions and their epilogue; the fix is SCATTER_REDUCE: add a structured pool-backward scatter lowering that fuses the avg-pool/add producer, masked scatter, sibling reductions, and BN-backward epilogue for this channels-last layout."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_0930bd38b7d1"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 192
LOW = 35
HIGH = 71
LOW_HW = LOW * LOW
HIGH_HW = HIGH * HIGH
NUMEL_LOW = N * C * LOW_HW
TOTAL_SPATIAL = N * HIGH_HW
NUMEL_HIGH = N * C * HIGH_HW
SCALE = 1.5497917079944455e-06



@triton.jit
def _scatter_unmasked_kernel(
    grad_pool_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    pool_offsets_ptr,
    convolution_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    where_ptr,
    NUMEL_LOW_: tl.constexpr,
    C_: tl.constexpr,
    LOW_: tl.constexpr,
    LOW_HW_: tl.constexpr,
    HIGH_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < NUMEL_LOW_

    c = offsets % C_
    spatial = offsets // C_
    ow = spatial % LOW_
    oh = (spatial // LOW_) % LOW_
    n = spatial // LOW_HW_

    raw_offset = tl.load(pool_offsets_ptr + offsets, mask=active, other=0).to(tl.int32)
    kh = raw_offset // 3
    kw = raw_offset - kh * 3
    h = oh * 2 + kh
    w = ow * 2 + kw
    high_offsets = ((n * HIGH_ + h) * HIGH_ + w) * C_ + c

    conv = tl.load(convolution_ptr + high_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine = (conv - mean) * rsqrt * weight + bias

    avg_grad = tl.zeros([BLOCK_ELEMS], dtype=tl.float32)
    # Match the CUDA channels-last avg_pool2d_backward kernel used by this repro.
    for dh in tl.static_range(-2, 1):
        src_h = oh + dh
        valid_h = active & (src_h >= 0) & (src_h < LOW_)
        src_h_safe = tl.where(valid_h, src_h, 0)
        for dw in tl.static_range(-2, 1):
            src_w = ow + dw
            valid = valid_h & (src_w >= 0) & (src_w < LOW_)
            src_w_safe = tl.where(valid, src_w, 0)
            src_offsets = ((n * LOW_ + src_h_safe) * LOW_ + src_w_safe) * C_ + c
            avg_grad += tl.load(grad_pool_ptr + src_offsets, mask=valid, other=0.0).to(tl.float32)
    avg_grad *= 0.1111111111111111

    add_value = tl.load(add0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    add_value += tl.load(add1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    add_value += tl.load(add2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    value = avg_grad + add_value
    tl.atomic_add(where_ptr + high_offsets, value, sem="relaxed", mask=active & (affine > 0.0))


@triton.jit
def _reduce_atomic_channels_last_kernel(
    where_ptr,
    convolution_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    full_ptr,
    sum1_ptr,
    sum2_ptr,
    TOTAL_SPATIAL_: tl.constexpr,
    C_: tl.constexpr,
    C_BLOCK_: tl.constexpr,
    BLOCK_SPATIAL_: tl.constexpr,
):
    spatial = tl.program_id(0) * BLOCK_SPATIAL_ + tl.arange(0, BLOCK_SPATIAL_)
    channels = tl.arange(0, C_BLOCK_)
    offsets = spatial[:, None] * C_ + channels[None, :]
    mask = (spatial[:, None] < TOTAL_SPATIAL_) & (channels[None, :] < C_)

    scatter_value = tl.load(where_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    conv = tl.load(convolution_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=channels < C_, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + channels, mask=channels < C_, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channels < C_, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channels < C_, other=0.0).to(tl.float32)
    full = tl.load(full_ptr).to(tl.float32)
    centered = conv - mean[None, :]
    affine = centered * rsqrt[None, :] * weight[None, :] + bias[None, :]
    where_self = tl.where(affine <= 0.0, full, scatter_value)

    partial1 = tl.sum(where_self, axis=0)
    partial2 = tl.sum(where_self * centered, axis=0)
    tl.atomic_add(sum1_ptr + channels, partial1, sem="relaxed", mask=channels < C_)
    tl.atomic_add(sum2_ptr + channels, partial2, sem="relaxed", mask=channels < C_)


@triton.jit
def _vector_out_kernel(
    sum2_ptr,
    rsqrt_ptr,
    vector_out_ptr,
    C_: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c = tl.arange(0, C_BLOCK)
    mask = c < C_
    sum2 = tl.load(sum2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(vector_out_ptr + c, sum2 * rsqrt, mask=mask)


@triton.jit
def _epilogue_inplace_kernel(
    where_ptr,
    convolution_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    full_ptr,
    sum1_ptr,
    sum2_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < NUMEL_

    c = offsets % C_
    scatter_value = tl.load(where_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    conv = tl.load(convolution_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    full = tl.load(full_ptr).to(tl.float32)

    centered = conv - mean
    affine = centered * rsqrt * weight + bias
    where_self = tl.where(affine <= 0.0, full, scatter_value)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = sum1 * SCALE_
    variance_term = sum2 * SCALE_ * rsqrt * rsqrt
    affine_scale = rsqrt * weight
    result = (where_self - centered * variance_term - mean_term) * affine_scale

    tl.store(where_ptr + offsets, result, mask=active)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) and not x.is_cuda else x for x in inputs)


def _check_shapes(
    getitem_442: torch.Tensor,
    getitem_12: torch.Tensor,
    getitem_451: torch.Tensor,
    getitem_457: torch.Tensor,
    getitem_460: torch.Tensor,
    getitem_13: torch.Tensor,
    convolution_4: torch.Tensor,
    getitem_11: torch.Tensor,
    rsqrt_4: torch.Tensor,
    primals_30: torch.Tensor,
    primals_31: torch.Tensor,
    full_default: torch.Tensor,
) -> None:
    low_shape = (N, C, LOW, LOW)
    high_shape = (N, C, HIGH, HIGH)
    assert getitem_442.shape == low_shape
    assert getitem_12.shape == low_shape
    assert getitem_451.shape == low_shape
    assert getitem_457.shape == low_shape
    assert getitem_460.shape == low_shape
    assert getitem_13.shape == low_shape
    assert convolution_4.shape == high_shape
    assert getitem_11.shape == (1, C, 1, 1)
    assert rsqrt_4.shape == (1, C, 1, 1)
    assert primals_30.shape == (C,)
    assert primals_31.shape == (C,)
    assert full_default.shape == ()
    assert getitem_13.stride(1) == 1
    assert convolution_4.stride(1) == 1


def oracle_scatter_reduce(
    getitem_442: torch.Tensor,
    getitem_12: torch.Tensor,
    getitem_451: torch.Tensor,
    getitem_457: torch.Tensor,
    getitem_460: torch.Tensor,
    getitem_13: torch.Tensor,
    convolution_4: torch.Tensor,
    getitem_11: torch.Tensor,
    rsqrt_4: torch.Tensor,
    primals_30: torch.Tensor,
    primals_31: torch.Tensor,
    full_default: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_shapes(
        getitem_442,
        getitem_12,
        getitem_451,
        getitem_457,
        getitem_460,
        getitem_13,
        convolution_4,
        getitem_11,
        rsqrt_4,
        primals_30,
        primals_31,
        full_default,
    )

    out = torch.empty(
        (N, C, HIGH, HIGH),
        device=getitem_442.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    block_elems = 512
    out.zero_()
    _scatter_unmasked_kernel[(triton.cdiv(NUMEL_LOW, block_elems),)](
        getitem_442,
        getitem_451,
        getitem_457,
        getitem_460,
        getitem_13,
        convolution_4,
        getitem_11,
        rsqrt_4,
        primals_30,
        primals_31,
        out,
        NUMEL_LOW_=NUMEL_LOW,
        C_=C,
        LOW_=LOW,
        LOW_HW_=LOW_HW,
        HIGH_=HIGH,
        BLOCK_ELEMS=block_elems,
        num_warps=8,
    )

    sum1 = torch.zeros((C,), device=getitem_442.device, dtype=torch.float32)
    sum2 = torch.zeros((C,), device=getitem_442.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=getitem_442.device, dtype=torch.float32)
    reduce_block_spatial = 32
    c_block = triton.next_power_of_2(C)
    _reduce_atomic_channels_last_kernel[(triton.cdiv(TOTAL_SPATIAL, reduce_block_spatial),)](
        out,
        convolution_4,
        getitem_11,
        rsqrt_4,
        primals_30,
        primals_31,
        full_default,
        sum1,
        sum2,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        C_=C,
        C_BLOCK_=c_block,
        BLOCK_SPATIAL_=reduce_block_spatial,
        num_warps=8,
    )
    _vector_out_kernel[(1,)](
        sum2,
        rsqrt_4,
        vector_out,
        C_=C,
        C_BLOCK=c_block,
        num_warps=8,
    )

    _epilogue_inplace_kernel[(triton.cdiv(NUMEL_HIGH, block_elems),)](
        out,
        convolution_4,
        getitem_11,
        rsqrt_4,
        primals_30,
        primals_31,
        full_default,
        sum1,
        sum2,
        NUMEL_=NUMEL_HIGH,
        C_=C,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=8,
    )

    return out, vector_out


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _compile_with_cd(model: torch.nn.Module, inputs: tuple[object, ...]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch({"coordinate_descent_tuning": True}):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


@oracle_impl(hardware="H100", shapes="(T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], i8, stride=(235200, 1, 6720, 192), gen=Index(9)), T([128, 192, 71, 71], f32, stride=(967872, 1, 13632, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([], f32), S([24576, 1225]), S([24576, 1225]), S([128, 192, 71, 71]))")
def oracle_forward(inputs):
    return oracle_scatter_reduce(*inputs)


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
