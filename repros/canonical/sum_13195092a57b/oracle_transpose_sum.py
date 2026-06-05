"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete `clone -> view -> permute` returned layout and the sibling `sum(dim=0).view([384])` output from one pass over the contiguous f32[8, 1500, 384] input, whereas Inductor treats the cloned/permute side output and the column reduction as separate scheduled work over the same flattened storage; Inductor cannot do this today because its scheduler does not form a full-scope multi-output reduction template whose epilogue both materializes the layout-changing side output and accumulates the compatible column sums; the fix is SCHEDULER_FUSION: add a dense copy-plus-reduction template for compatible materialized side outputs and sibling reductions, falling back to BANDWIDTH_BOUND when the fused form matches the memory-copy floor."""
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


N_COLS = 384
BLOCK_M = 256
BLOCK_N = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _transpose_clone_sum_kernel(
        x_ptr,
        out_perm_storage_ptr,
        out_sum_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)[:, None]
        cols = pid_n * BLOCK_N_ + tl.arange(0, BLOCK_N_)[None, :]
        mask = (rows < M) & (cols < N)
        offsets = rows * N + cols

        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_perm_storage_ptr + offsets, values, mask=mask)
        sums = tl.sum(values.to(tl.float32), axis=0)
        sum_cols = pid_n * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        tl.atomic_add(out_sum_ptr + sum_cols, sums, sem="relaxed", mask=sum_cols < N)


def _as_shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in value)
    except TypeError as exc:
        raise TypeError(f"expected shape-like value, got {type(value)!r}") from exc


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, int, int]:
    if len(inputs) != 3:
        raise RuntimeError(f"expected three inputs, got {len(inputs)}")

    arg25_1, shape_param_0, shape_param_1 = inputs
    if not isinstance(arg25_1, torch.Tensor):
        raise RuntimeError(f"expected tensor input, got {type(arg25_1)!r}")
    if arg25_1.dtype != torch.float32:
        raise RuntimeError(f"expected float32 input, got {arg25_1.dtype}")
    if not arg25_1.is_cuda:
        raise RuntimeError("CUDA tensor input is required")
    if not arg25_1.is_contiguous():
        raise RuntimeError(f"expected contiguous input, got stride={arg25_1.stride()}")

    flat_shape = _as_shape(shape_param_0)
    sum_shape = _as_shape(shape_param_1)
    if len(flat_shape) != 2:
        raise RuntimeError(f"expected rank-2 flattened shape, got {flat_shape}")
    if flat_shape[1] != N_COLS:
        raise RuntimeError(f"expected flattened shape with {N_COLS} columns, got {flat_shape}")
    if sum_shape != (N_COLS,):
        raise RuntimeError(f"expected sum output shape ({N_COLS},), got {sum_shape}")
    if arg25_1.numel() != flat_shape[0] * flat_shape[1]:
        raise RuntimeError(
            f"input numel {arg25_1.numel()} does not match flattened shape {flat_shape}"
        )

    return arg25_1, flat_shape[0], flat_shape[1]


def oracle_forward(inputs):
    """Run the full-scope oracle computation.

    The eager graph clones the input, views it as [12000, 384], returns the
    permuted [384, 12000] view of that clone, and returns the column sum. The
    returned permute has strides [1, 384], so its physical storage order is the
    same flattened order as the input.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    arg25_1, m_rows, n_cols = _validate_inputs(inputs)
    out_perm = torch.empty_strided(
        (n_cols, m_rows),
        (1, n_cols),
        device=arg25_1.device,
        dtype=arg25_1.dtype,
    )
    out_sum = torch.empty_strided((n_cols,), (1,), device=arg25_1.device, dtype=arg25_1.dtype)
    out_sum.zero_()
    _transpose_clone_sum_kernel[(triton.cdiv(m_rows, BLOCK_M), triton.cdiv(n_cols, BLOCK_N))](
        arg25_1,
        out_perm,
        out_sum,
        M=m_rows,
        N=n_cols,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        num_warps=8,
    )
    return out_perm, out_sum


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
