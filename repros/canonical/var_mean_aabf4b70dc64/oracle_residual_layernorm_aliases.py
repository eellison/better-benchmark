"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete M2M100 fp32 residual-add LayerNorm scope, including the [8192,1024] to [64,128,1024] view, residual add, population var_mean over hidden size 1024, eps=1e-5 rsqrt, fp32 affine scale and bias, and twenty-four returned [8192,1024] alias views from one Triton row kernel, whereas Inductor lowers the residual add, generic var_mean reduction, normalization epilogue, affine pointwise ops, and sibling view returns through its general reduction scheduling path; Inductor cannot do this today because its normalization pattern/codegen does not recognize fixed-width LayerNorm with residual-add input and many alias view outputs as a single reusable lowering; the fix is NEW_PATTERN: add a guarded residual-add LayerNorm alias template that folds the residual producer, two-pass fp32 var_mean reduction, affine epilogue, and multi-view return contract into specialized codegen."""
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

BATCH = 64
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 1024
BASE_SHAPE = (BATCH, SEQ, HIDDEN)
BASE_STRIDE = (SEQ * HIDDEN, HIDDEN, 1)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_COUNT = 24
EPS = 1.0e-5

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
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        values = (
            tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        mean = tl.sum(tl.where(mask, values, 0.0), axis=0) / hidden
        centered = values - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        inv_std = tl.rsqrt(var + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = centered * inv_std * weight + bias

        tl.store(out_ptr + offsets, out, mask=mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
    if len(inputs) != 4 + 1 + OUTPUT_COUNT:
        raise ValueError(f"{REPRO_ID} expects 29 inputs, got {len(inputs)}")

    mm = _require_tensor("addmm_71", inputs[0], OUTPUT_SHAPE, torch.float32)
    residual = _require_tensor("add_84", inputs[1], BASE_SHAPE, torch.float32)
    weight = _require_tensor("arg196_1", inputs[2], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg197_1", inputs[3], (HIDDEN,), torch.float32)

    if residual.device != mm.device or weight.device != mm.device or bias.device != mm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    _require_shape("_shape_param_0", inputs[4], BASE_SHAPE)
    output_shapes = tuple(
        _require_shape(f"_shape_param_{index}", inputs[4 + index], OUTPUT_SHAPE)
        for index in range(1, OUTPUT_COUNT + 1)
    )
    return mm, residual, weight, bias, output_shapes


@oracle_impl(hardware="H100", shapes="(T([8192, 1024], f32), T([64, 128, 1024], f32), T([1024], f32), T([1024], f32), S([64, 128, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]))")
def oracle_forward(inputs):
    """Run the full residual-add LayerNorm repro computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same twenty-four fp32 [8192,1024] views. The returned tensors alias one
    [64,128,1024] contiguous result storage, matching the eager view contract.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_layernorm_aliases.py")

    mm, residual, weight, bias, output_shapes = _validate_inputs(inputs)
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )

    _residual_layernorm_kernel[(ROWS,)](
        mm,
        residual,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=HIDDEN,
        num_warps=4,
        num_stages=3,
    )

    return tuple(base.view(shape) for shape in output_shapes)


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
