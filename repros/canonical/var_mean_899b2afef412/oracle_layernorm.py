"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete f32 affine LayerNorm scope as one shape-specialized Triton row-reduction kernel, including population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt, broadcast scale/bias, and the metadata-only final `[B*S, H]` view, whereas Inductor already emits one fused normalization kernel but uses its generic var_mean row schedule with extra large-hidden indexing and reduction overhead for these sequence-to-flat views; Inductor cannot do this today because the scheduler/codegen path does not select a guarded large-hidden affine-LayerNorm row tile for `var_mean` followed by broadcast affine and a view-only epilogue; the fix is SCHEDULER_FUSION: add a benchmark-gated large-hidden affine LayerNorm schedule that specializes hidden-size tiling and writes the final viewed layout directly."""
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
EPS = 1.0e-5

if triton is not None:

    @triton.jit
    def _layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        offsets = row * hidden + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden

        centered = x - mean
        centered_for_reduce = tl.where(mask, centered, 0.0)
        variance = tl.sum(centered_for_reduce * centered_for_reduce, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)


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
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    x, weight, bias, output_shape_value = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"arg2_1 must be a tensor, got {type(x).__name__}")
    if x.ndim != 3:
        raise ValueError(f"arg2_1 must be rank 3, got shape={tuple(x.shape)}")

    batch, seq_len, hidden = tuple(int(dim) for dim in x.shape)
    rows = batch * seq_len
    if hidden not in (1024, 2560):
        raise ValueError(f"unexpected hidden size {hidden}; expected 1024 or 2560")

    x_t = _require_tensor("arg2_1", x, (batch, seq_len, hidden), torch.float32)
    weight_t = _require_tensor("arg0_1", weight, (hidden,), torch.float32)
    bias_t = _require_tensor("arg1_1", bias, (hidden,), torch.float32)

    if weight_t.device != x_t.device or bias_t.device != x_t.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    output_shape = _shape_tuple(output_shape_value)
    if output_shape != (rows, hidden):
        raise ValueError(
            f"unexpected final view shape parameter {output_shape_value!r}; "
            f"expected {(rows, hidden)}"
        )

    return x_t, weight_t, bias_t, output_shape


@oracle_impl(hardware="H100", shapes="(T([32, 128, 2560], f32), T([2560], f32), T([2560], f32), S([4096, 2560]))")
def oracle_forward(inputs):
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
        raise RuntimeError("Triton is required for oracle_layernorm.py")

    x, weight, bias, output_shape = _validate_inputs(inputs)
    hidden = int(x.shape[2])
    rows = int(output_shape[0])
    block_h = triton.next_power_of_2(hidden)
    out = torch.empty_strided(
        output_shape,
        (hidden, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _layernorm_kernel[(rows,)](
        x,
        weight,
        bias,
        out,
        hidden=hidden,
        eps=EPS,
        BLOCK_H=block_h,
        num_warps=8,
        num_stages=3,
    )
    return out


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
