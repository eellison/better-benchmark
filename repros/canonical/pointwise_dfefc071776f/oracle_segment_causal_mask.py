"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full six-output segmented causal mask scope with one direct Triton pointwise kernel that writes six independent `[B,1,S,S]` bases and returns the same expanded zero-head-stride views as eager, whereas Inductor already fuses the captured iota/index/compare/where/expand graph into one pointwise kernel for the full scope; Inductor cannot materially improve this today because the required six independent output storages dominate runtime once the small `cumsum` tensor is cache-resident; the fix is BANDWIDTH_BOUND: treat this repro as at floor unless generic pointwise store bandwidth or output alias legality changes."""
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
    def _segment_causal_mask_kernel(
        cumsum_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        out4_ptr,
        out5_ptr,
        N_ELEMENTS: tl.constexpr,
        S: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS

        col = offsets % S
        tmp = offsets // S
        row = tmp % S
        batch = tmp // S

        row_value = tl.load(cumsum_ptr + batch * S + row, mask=mask, other=0)
        col_value = tl.load(cumsum_ptr + batch * S + col, mask=mask, other=-1)
        keep = (col <= row) & (row_value == col_value)
        value = tl.where(keep, 0.0, -float("inf"))

        tl.store(out0_ptr + offsets, value, mask=mask)
        tl.store(out1_ptr + offsets, value, mask=mask)
        tl.store(out2_ptr + offsets, value, mask=mask)
        tl.store(out3_ptr + offsets, value, mask=mask)
        tl.store(out4_ptr + offsets, value, mask=mask)
        tl.store(out5_ptr + offsets, value, mask=mask)


def _shape_tuple(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_expand_shape(name: str, value: Any, base_shape: tuple[int, ...]) -> tuple[int, ...]:
    dims = list(_shape_tuple(name, value))
    if len(dims) != len(base_shape):
        raise ValueError(f"{name} rank must be {len(base_shape)}, got {len(dims)}")
    for idx, dim in enumerate(dims):
        if dim == -1:
            dims[idx] = base_shape[idx]
        elif dim != base_shape[idx] and base_shape[idx] != 1:
            raise ValueError(f"{name} cannot expand base shape {base_shape} to {tuple(dims)}")
    return tuple(dims)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[tuple[int, int, int, int], ...]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects eight inputs, got {len(inputs)}")

    cumsum = inputs[0]
    if not isinstance(cumsum, torch.Tensor):
        raise TypeError("input 0 must be a torch.Tensor")
    if not cumsum.is_cuda:
        raise TypeError("input 0 must be a CUDA tensor")
    if cumsum.dtype is not torch.int64:
        raise TypeError(f"input 0 must be torch.int64, got {cumsum.dtype}")
    if cumsum.ndim != 2:
        raise ValueError(f"input 0 must be rank-2, got shape={tuple(cumsum.shape)}")
    if not cumsum.is_contiguous():
        raise ValueError(f"input 0 must be contiguous, got stride={tuple(cumsum.stride())}")

    batch = int(cumsum.shape[0])
    seq = int(cumsum.shape[1])
    base_shape = _resolve_expand_shape("shape_param_0", inputs[1], (batch, 1, seq, seq))
    if base_shape != (batch, 1, seq, seq):
        raise ValueError(f"shape_param_0 must preserve the singleton head dimension, got {base_shape}")

    output_shapes = tuple(
        _resolve_expand_shape(f"shape_param_{idx}", shape_param, base_shape)
        for idx, shape_param in enumerate(inputs[2:], start=1)
    )
    if len(output_shapes) != 6:
        raise ValueError(f"{REPRO_ID} expects six output shape params, got {len(output_shapes)}")

    return cumsum, base_shape, output_shapes


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
        raise RuntimeError("Triton is required for this oracle")

    cumsum, base_shape, output_shapes = _validate_inputs(inputs)
    bases = tuple(torch.empty(base_shape, dtype=torch.float32, device=cumsum.device) for _ in range(6))

    n_elements = bases[0].numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _segment_causal_mask_kernel[grid](
        cumsum,
        bases[0],
        bases[1],
        bases[2],
        bases[3],
        bases[4],
        bases[5],
        N_ELEMENTS=n_elements,
        S=base_shape[2],
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=4,
    )

    return tuple(base.expand(shape) for base, shape in zip(bases, output_shapes))


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
