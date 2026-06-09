"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT ReLU plus correction=0 fp32 variance/mean over the 512-wide hidden dimension, eps-before-libdevice.rsqrt normalization, affine multiply/add, and returned non-contiguous `[512,32768]` permute layout in one Triton kernel, whereas Inductor already emits the same single persistent inner-reduction kernel and returns the same reinterpret stride; Inductor cannot materially improve this repro through local fusion because there is no remaining avoidable intermediate or launch in the captured full scope; the fix is BANDWIDTH_BOUND: record this as a full-scope norm-template floor unless broader reduction-template codegen improves both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


ROWS = 32768
HIDDEN = 512
INPUT_SHAPE = (ROWS, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
VIEW_SHAPE = (256, 128, HIDDEN)
OUTPUT_SHAPE = (HIDDEN, ROWS)
OUTPUT_STRIDE = (1, HIDDEN)
EPS = 1.0e-12


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice

    triton_helpers.set_driver_to_gpu()

    @triton.jit
    def _mobilebert_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps_value: tl.constexpr,
        block_m: tl.constexpr,
    ):
        row = tl.program_id(0) * block_m + tl.arange(0, block_m)[:, None]
        col = tl.arange(0, hidden)[None, :]
        offsets = row * hidden + col

        x = tl.load(x_ptr + offsets, None, eviction_policy="evict_first")
        weight = tl.load(weight_ptr + col, None, eviction_policy="evict_last")
        bias = tl.load(bias_ptr + col, None, eviction_policy="evict_last")

        zero = tl.full([1, 1], 0, tl.int32)
        relu = triton_helpers.maximum(zero, x)
        relu_b = tl.broadcast_to(relu, [block_m, hidden])

        sum_relu = tl.sum(relu_b, 1)[:, None].to(tl.float32)
        denom_i = tl.full([1, 1], hidden, tl.int32)
        mean = sum_relu / denom_i.to(tl.float32)

        centered_for_var = relu_b - mean
        sum_sq = tl.sum(tl.broadcast_to(centered_for_var * centered_for_var, [block_m, hidden]), 1)[:, None].to(tl.float32)
        denom_f = tl.full([1, 1], 512.0, tl.float32)
        var = sum_sq / denom_f
        eps = tl.full([1, 1], eps_value, tl.float32)
        inv_std = libdevice.rsqrt(var + eps)

        centered = relu - mean
        normalized = centered * inv_std
        scaled = normalized * weight
        out = scaled + bias

        tl.store(out_ptr + offsets, out, None)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _expect_f32_cuda_1d(name: str, value: Any, size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (size,):
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {(size,)}")
    if tuple(value.stride()) != (1,):
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected (1,)")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"addmm_361 must be a tensor, got {type(x)!r}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"addmm_361 has shape {tuple(x.shape)}, expected {INPUT_SHAPE}")
    if tuple(x.stride()) != (HIDDEN, 1):
        raise ValueError(f"addmm_361 has stride {tuple(x.stride())}, expected {(HIDDEN, 1)}")
    if x.dtype != torch.float32:
        raise TypeError(f"addmm_361 has dtype {x.dtype}, expected torch.float32")
    if not x.is_cuda:
        raise RuntimeError("addmm_361 must be a CUDA tensor for this Triton oracle")

    weight_t = _expect_f32_cuda_1d("arg1115_1", weight, HIDDEN)
    bias_t = _expect_f32_cuda_1d("arg1116_1", bias, HIDDEN)
    if weight_t.device != x.device or bias_t.device != x.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != INPUT_SHAPE:
        raise ValueError(f"unexpected second view shape parameter: {shape1!r}")

    return x, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([32768, 512], f32), T([512], f32), T([512], f32), S([256, 128, 512]), S([32768, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope and return the same non-contiguous view."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mobilebert_layernorm.py")

    x, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    _mobilebert_layernorm_kernel[(triton.cdiv(ROWS, 1),)](
        x,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps_value=EPS,
        block_m=1,
    )
    return out


def _check_output_layout(inputs: list[Any] | tuple[Any, ...], instance) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        oracle = oracle_forward(inputs)
    ok = (
        tuple(oracle.shape) == tuple(eager.shape)
        and tuple(oracle.stride()) == tuple(eager.stride())
        and not oracle.is_contiguous()
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"  layout: {status} "
        f"(shape={list(oracle.shape)} stride={tuple(oracle.stride())} "
        f"eager_stride={tuple(eager.stride())} contiguous={oracle.is_contiguous()})"
    )
    return ok


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

    inputs = get_inputs()
    instance = get_repro_instance()

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
        layout_ok = _check_output_layout(inputs, instance)
        ok = ok and layout_ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
