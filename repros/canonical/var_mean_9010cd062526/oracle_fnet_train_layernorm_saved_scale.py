"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete FNet train real-select residual LayerNorm scope in one shape-specialized Triton row kernel, including view_as_real_11[..., 0], residual add, correction=0 hidden-size-768 var_mean, eps=1e-12 rsqrt, affine output view, and rsqrt / 768 side output, whereas Inductor currently emits one fused generic var_mean reduction kernel for the same select/add/affine/saved-scale graph; Inductor cannot do this today because correction=0 var_mean lowering keeps generic Welford reduction/codegen instead of selecting a fixed-hidden LayerNorm algebra that reuses the row tile for the affine epilogue and saved inverse-scale store; the fix is ALGEBRAIC_ELIMINATION: add a guarded fixed-hidden correction=0 LayerNorm lowering that replaces generic Welford bookkeeping with direct mean and centered-variance reductions while preserving the side output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 32
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
REAL_IMAG = 2
BLOCK_H = 1024
EPS = 1.0e-12
INPUT_COMPLEX_SHAPE = (BATCH, SEQ_LEN, HIDDEN, REAL_IMAG)
RESIDUAL_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for the Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    view_as_real = _require_tensor(
        "view_as_real_11", inputs[0], INPUT_COMPLEX_SHAPE, torch.float32
    )
    residual = _require_tensor("add_91", inputs[1], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg98_1", inputs[2], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg99_1", inputs[3], (HIDDEN,), torch.float32)
    output_shape = _shape_tuple(inputs[4])

    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {inputs[4]!r}")

    expected_view_stride = (SEQ_LEN * HIDDEN * REAL_IMAG, HIDDEN * REAL_IMAG, REAL_IMAG, 1)
    expected_residual_stride = (SEQ_LEN * HIDDEN, HIDDEN, 1)
    if tuple(view_as_real.stride()) != expected_view_stride:
        raise ValueError(
            f"view_as_real_11 stride {view_as_real.stride()} != {expected_view_stride}"
        )
    if tuple(residual.stride()) != expected_residual_stride:
        raise ValueError(f"add_91 stride {residual.stride()} != {expected_residual_stride}")
    if tuple(weight.stride()) != (1,) or tuple(bias.stride()) != (1,):
        raise ValueError(
            f"affine inputs must be contiguous, got weight={weight.stride()} bias={bias.stride()}"
        )

    device = view_as_real.device
    for name, tensor in (("add_91", residual), ("arg98_1", weight), ("arg99_1", bias)):
        if tensor.device != device:
            raise ValueError(f"{name} device {tensor.device} != {device}")

    return view_as_real, residual, weight, bias, output_shape


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _fnet_train_layernorm_saved_scale_kernel(
        view_as_real_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        side_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        dense_offsets = row * hidden + cols
        real_offsets = dense_offsets * 2

        real_part = tl.load(
            view_as_real_ptr + real_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + dense_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = real_part + residual
        x = tl.where(mask, x, 0.0)

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        centered_for_reduce = tl.where(mask, centered, 0.0)
        variance = tl.sum(centered_for_reduce * centered_for_reduce, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(output_ptr + dense_offsets, y, mask=mask)
        tl.store(side_ptr + row, invstd / hidden)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward computation with a fused row LayerNorm kernel.

    SCOPE INVARIANT: accepts the same five inputs as Repro.forward() and returns
    the same `(f32[16384,768], f32[32,512,1])` output structure and strides.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    view_as_real, residual, weight, bias, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        (HIDDEN, 1),
        device=view_as_real.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=view_as_real.device,
        dtype=torch.float32,
    )
    _fnet_train_layernorm_saved_scale_kernel[(ROWS,)](
        view_as_real,
        residual,
        weight,
        bias,
        output,
        side,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=8,
        num_stages=3,
    )
    return output, side


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    actual_outputs = actual if isinstance(actual, tuple) else (actual,)
    ok = len(expected_outputs) == len(actual_outputs)
    if ok:
        ok = all(
            tuple(o.shape) == tuple(e.shape)
            and o.dtype == e.dtype
            and tuple(o.stride()) == tuple(e.stride())
            for e, o in zip(expected_outputs, actual_outputs)
        )

    for index, (eager, oracle) in enumerate(zip(expected_outputs, actual_outputs)):
        print(
            f"  output {index} layout: "
            f"{'PASS' if tuple(oracle.stride()) == tuple(eager.stride()) else 'FAIL'} "
            f"(expected_stride={eager.stride()}, oracle_stride={oracle.stride()})"
        )
    return ok


# --- CLI entry point ---
def main() -> None:
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
        ok = _check_layout(instance, inputs) and ok
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
