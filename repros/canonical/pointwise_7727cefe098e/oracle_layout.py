"""
Full-scope oracle for pointwise_7727cefe098e.

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
complete batchnorm-like pointwise expression, ReLU, `[2, 1]` max-pool with
int8 offsets, and final eager-layout permuted f32 view in one Triton kernel,
whereas Inductor emits separate generated kernels for the pointwise producer
and the low-memory max-pool-with-offsets consumer; Inductor cannot do this
today because scheduler fusion does not sink the elementwise producer into the
structured pooling kernel while also writing the pooled values in the final
view/permute backing layout, so the fix is SCHEDULER_FUSION: allow guarded
producer fusion into low-memory max-pool-with-offsets and direct layout-aware
multi-output stores for the offsets plus pooled value view.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CHANNELS = 512
HEIGHT = 2
WIDTH = 32
OUT_NUMEL = CHANNELS * WIDTH
CONV_SHAPE = (1, CHANNELS, HEIGHT, WIDTH)
PARAM_SHAPE = (CHANNELS,)
OFFSETS_SHAPE = (1, CHANNELS, 1, WIDTH)
OFFSETS_STRIDE = (CHANNELS * WIDTH, WIDTH, WIDTH, 1)
POOLED_BASE_SHAPE = (1, CHANNELS, WIDTH)
POOLED_BASE_STRIDE = (CHANNELS * WIDTH, WIDTH, 1)
PERMUTED_SHAPE = (1, WIDTH, CHANNELS)
PERMUTED_STRIDE = (CHANNELS * WIDTH, 1, WIDTH)
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_f32(x):
        # `where` preserves NaNs like aten.relu, while clamping negatives to +0.
        return tl.where(x <= 0.0, 0.0, x)


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL"],
    )
    @triton.jit
    def _bn_relu_pool_offsets_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        offsets_ptr,
        pooled_ptr,
        TOTAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = linear < TOTAL
        channel = linear // 32
        col = linear - channel * 32

        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        inv_std = 1.0 / tl.sqrt(var + 0.00001)
        scale = inv_std * weight

        conv_base = channel * 64 + col
        x0 = tl.load(conv_ptr + conv_base, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(conv_ptr + conv_base + 32, mask=mask, other=0.0).to(tl.float32)

        y0 = _relu_f32((x0 - mean) * scale + bias)
        y1 = _relu_f32((x1 - mean) * scale + bias)

        take_1 = (y1 > y0) | (y1 != y1)
        pooled = tl.where(take_1, y1, y0)
        offset = tl.where(take_1, 1, 0)

        tl.store(offsets_ptr + linear, offset, mask=mask)
        tl.store(pooled_ptr + linear, pooled, mask=mask)


def oracle_forward(inputs):
    """Run the full Repro.forward scope with one Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    arg75_1, convolution_12, arg76_1, arg77_1, arg78_1, shape_param = inputs
    tensor_inputs = (arg75_1, convolution_12, arg76_1, arg77_1, arg78_1)
    for index, tensor in enumerate(tensor_inputs):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"input {index} must be a tensor, got {type(tensor)!r}")
        if tensor.dtype is not torch.float32:
            raise ValueError(f"input {index} must be torch.float32, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"input {index} must be a CUDA tensor")

    for name, tensor in (
        ("arg75_1", arg75_1),
        ("arg76_1", arg76_1),
        ("arg77_1", arg77_1),
        ("arg78_1", arg78_1),
    ):
        if tuple(tensor.shape) != PARAM_SHAPE or tensor.stride() != (1,):
            raise ValueError(f"{name} must have shape {PARAM_SHAPE} and stride (1,), got {tuple(tensor.shape)} {tensor.stride()}")
    if tuple(convolution_12.shape) != CONV_SHAPE or not convolution_12.is_contiguous():
        raise ValueError(
            f"convolution_12 must be contiguous shape {CONV_SHAPE}, "
            f"got {tuple(convolution_12.shape)} stride={convolution_12.stride()}"
        )
    if list(shape_param) != [-1, CHANNELS, WIDTH]:
        raise ValueError(f"unexpected shape parameter: {shape_param!r}")

    offsets = torch.empty_strided(
        OFFSETS_SHAPE,
        OFFSETS_STRIDE,
        device=convolution_12.device,
        dtype=torch.int8,
    )
    pooled_base = torch.empty_strided(
        POOLED_BASE_SHAPE,
        POOLED_BASE_STRIDE,
        device=convolution_12.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(OUT_NUMEL, meta["BLOCK"]),)
    _bn_relu_pool_offsets_kernel[grid](
        arg75_1,
        convolution_12,
        arg76_1,
        arg77_1,
        arg78_1,
        offsets,
        pooled_base,
        TOTAL=OUT_NUMEL,
    )
    return offsets, pooled_base.permute(0, 2, 1)


def _max_abs_diff(got: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (got.float() - ref.float()).abs()
    finite = diff[torch.isfinite(diff)]
    if finite.numel() == 0:
        return math.nan
    return float(finite.max().item())


def _check_oracle_equal_nan(instance, inputs, atol: float, rtol: float) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(eager, tuple):
        eager = (eager,)
    if not isinstance(actual, tuple):
        actual = (actual,)
    if len(actual) != len(eager):
        print(f"  SCOPE_MISMATCH: oracle produces {len(actual)} outputs, eager produces {len(eager)}")
        return False

    ok = True
    for index, (got, ref) in enumerate(zip(actual, eager)):
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        if ref.is_floating_point():
            value_ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol, equal_nan=True)
            max_diff = _max_abs_diff(got, ref)
            nan_match = torch.equal(torch.isnan(got), torch.isnan(ref))
            print(
                f"  output {index}: {'PASS' if shape_ok and dtype_ok and stride_ok and value_ok and nan_match else 'FAIL'} "
                f"(shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
                f"max_diff={max_diff:.2e} nan_match={nan_match})"
            )
            ok = ok and shape_ok and dtype_ok and stride_ok and value_ok and nan_match
        else:
            value_ok = torch.equal(got, ref)
            print(
                f"  output {index}: {'PASS' if shape_ok and dtype_ok and stride_ok and value_ok else 'FAIL'} "
                f"(exact, shape={list(got.shape)} dtype={got.dtype} stride={got.stride()})"
            )
            ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Accepted for template CLI compatibility; this oracle checks all outputs.",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_oracle_equal_nan(instance, inputs, atol=args.atol, rtol=args.rtol)
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
