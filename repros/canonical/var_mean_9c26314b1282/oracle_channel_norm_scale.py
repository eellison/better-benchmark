"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full captured NFNet per-channel var_mean normalization and scalar gain in one shape-specialized Triton program per channel, using raw sum/sum-of-squares moments for correction=0 variance and reusing the loaded 1152-element channel tile for the normalized output store, whereas coordinate-descent Inductor emits one fused generic var_mean kernel that keeps Welford reduction state and reloads the activation for the normalize epilogue; Inductor cannot do this today because its var_mean reduction lowering does not have a guarded raw-moment small-K normalization template that can trade Welford bookkeeping and the second activation load for the captured tolerance envelope; the fix is ALGEBRAIC_ELIMINATION: add a correction=0 f32 var_mean+normalize lowering that uses sum and sumsq moments and retains the reduction tile for sibling invstd mean and normalized-output stores when shapes and accuracy policy allow it."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
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


CHANNELS = 768
INNER_SIZE = 128 * 3 * 3
EPS = 1.0e-5
GAIN_SCALE = 0.02946278254943948
OUT_SHAPE = (CHANNELS, 128, 3, 3)
OUT_STRIDE = (INNER_SIZE, 9, 3, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
MEAN_SHAPE = (1, CHANNELS, 1)
MEAN_STRIDE = (CHANNELS, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_K": 2048}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_K": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_K": 2048}, num_warps=8, num_stages=4),
        ],
        key=["k_size"],
    )
    @triton.jit
    def _channel_var_mean_norm_kernel(
        x_ptr,
        gain_ptr,
        invstd_ptr,
        out_ptr,
        mean_ptr,
        k_size: tl.constexpr,
        eps: tl.constexpr,
        gain_scale: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < k_size
        base = channel * k_size + offsets

        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / k_size
        var = sum_x2 / k_size - mean * mean
        invstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)
        gain = tl.load(gain_ptr + channel).to(tl.float32) * gain_scale

        y = (x - mean) * invstd * gain
        tl.store(invstd_ptr + channel, invstd)
        tl.store(mean_ptr + channel, mean)
        tl.store(out_ptr + base, y, mask=mask)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    x, gain, view_shape, out_shape = inputs
    x = _expect_tensor("arg218_1", x, OUT_SHAPE, OUT_STRIDE)
    gain = _expect_tensor("arg219_1", gain, (CHANNELS, 1, 1, 1), (1, 1, 1, 1))
    if x.device != gain.device:
        raise ValueError(f"tensor inputs must be on the same device, got {x.device} and {gain.device}")
    if tuple(view_shape) != (1, CHANNELS, INNER_SIZE):
        raise ValueError(f"unexpected view shape parameter: {view_shape!r}")
    if tuple(out_shape) != OUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter: {out_shape!r}")
    return x, gain


@oracle_impl(hardware="H100", shapes="(T([768, 128, 3, 3], f32), T([768, 1, 1, 1], f32), S([1, 768, 1152]), S([768, 128, 3, 3]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, gain = _validate_inputs(inputs)
    invstd = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=x.dtype)
    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x.device, dtype=x.dtype)
    mean = torch.empty_strided(MEAN_SHAPE, MEAN_STRIDE, device=x.device, dtype=x.dtype)

    _channel_var_mean_norm_kernel[(CHANNELS,)](
        x,
        gain,
        invstd,
        out,
        mean,
        k_size=INNER_SIZE,
        eps=EPS,
        gain_scale=GAIN_SCALE,
    )
    return invstd, out, mean


def _tensor_outputs(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(_tensor_outputs(item))
        return result
    return []


def _check_layout_and_aliases(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_tensors = _tensor_outputs(expected)
    actual_tensors = _tensor_outputs(actual)
    if len(expected_tensors) != len(actual_tensors):
        print(
            "  layout: FAIL "
            f"(output count eager={len(expected_tensors)} oracle={len(actual_tensors)})"
        )
        return False

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected_tensors, actual_tensors)):
        output_ok = (
            expected_tensor.shape == actual_tensor.shape
            and expected_tensor.dtype == actual_tensor.dtype
            and expected_tensor.stride() == actual_tensor.stride()
        )
        ok = ok and output_ok
        print(
            f"  output {index} layout: {'PASS' if output_ok else 'FAIL'} "
            f"(expected_shape={tuple(expected_tensor.shape)}, oracle_shape={tuple(actual_tensor.shape)}, "
            f"expected_dtype={expected_tensor.dtype}, oracle_dtype={actual_tensor.dtype}, "
            f"expected_stride={expected_tensor.stride()}, oracle_stride={actual_tensor.stride()})"
        )

    for left in range(len(expected_tensors)):
        for right in range(left + 1, len(expected_tensors)):
            expected_alias = expected_tensors[left].data_ptr() == expected_tensors[right].data_ptr()
            actual_alias = actual_tensors[left].data_ptr() == actual_tensors[right].data_ptr()
            pair_ok = expected_alias == actual_alias
            ok = ok and pair_ok
            print(
                f"  alias {left}/{right}: {'PASS' if pair_ok else 'FAIL'} "
                f"(expected={expected_alias}, oracle={actual_alias})"
            )
    return ok


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
        ok = ok and _check_layout_and_aliases(instance, inputs)
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
