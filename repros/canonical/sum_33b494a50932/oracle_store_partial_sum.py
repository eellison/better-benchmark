"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full exact-GELU derivative producer once, writes the required base buffer for the returned transposed view, and accumulates Inductor-shaped 128-row column partials in that same Triton program before a 64-partial final reduction, whereas Inductor currently emits a separate pointwise materialization followed by two reduction kernels that reread the materialized producer; Inductor cannot do this today because its scheduler does not form a multi-output producer-store plus reduction-partial node for a value that is both returned as a view and reduced; the fix is SCHEDULER_FUSION: teach multi-output reduction scheduling to generate a fused store-and-partial-reduce template while preserving the aliasing output view."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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
    has_stochastic_ops,
)

M = 8192
N = 3072
ROW_BLOCK = 128
ROW_BLOCKS = M // ROW_BLOCK
FINAL_BLOCK_N = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=3),
        ],
        key=["n_cols"],
    )
    @triton.jit
    def _store_and_partial_sum_kernel(
        mm_ptr,
        x_ptr,
        base_out_ptr,
        partials_ptr,
        n_cols: tl.constexpr,
        ROW_BLOCK_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        col_block = tl.program_id(1)

        rows = row_block * ROW_BLOCK_ + tl.arange(0, ROW_BLOCK_)[:, None]
        cols_1d = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        cols = cols_1d[None, :]
        col_mask = cols_1d < n_cols
        offsets = rows * n_cols + cols

        tmp0 = tl.load(mm_ptr + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32)
        tmp1 = tl.load(x_ptr + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32)
        tmp2 = tmp1 * 0.7071067811865476
        tmp3 = libdevice.erf(tmp2)
        tmp4 = tmp3 + 1.0
        tmp5 = tmp4 * 0.5
        tmp6 = tmp1 * tmp1
        tmp7 = tmp6 * -0.5
        tmp8 = libdevice.exp(tmp7)
        tmp9 = tmp8 * 0.3989422804014327
        tmp10 = tmp1 * tmp9
        tmp11 = tmp5 + tmp10
        tmp12 = tmp0 * tmp11

        tl.store(base_out_ptr + offsets, tmp12, mask=col_mask[None, :])
        partial = tl.sum(tl.where(col_mask[None, :], tmp12, 0.0), axis=0)
        tl.store(partials_ptr + row_block * n_cols + cols_1d, partial, mask=col_mask)

    @triton.jit
    def _final_sum_kernel(
        partials_ptr,
        sum_out_ptr,
        n_cols: tl.constexpr,
        ROW_BLOCKS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        cols_1d = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        cols = cols_1d[:, None]
        chunks = tl.arange(0, ROW_BLOCKS_)[None, :]
        mask = cols_1d < n_cols
        vals = tl.load(partials_ptr + chunks * n_cols + cols, mask=mask[:, None], other=0.0)
        total = tl.sum(vals, axis=1)
        tl.store(sum_out_ptr + cols_1d, total, mask=mask)


def _expect_tensor(name: str, value: Any, shape: tuple[int, int]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if tuple(value.stride()) != (N, 1):
        raise ValueError(f"{name} stride {tuple(value.stride())} != {(N, 1)}")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} storage offset {value.storage_offset()} != 0")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_store_partial_sum.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm = _expect_tensor("mm_132", inputs[0], (M, N))
    x = _expect_tensor("arg106_1", inputs[1], (M, N))
    if list(inputs[2]) != [8, 1024, 3072]:
        raise ValueError(f"unexpected first view shape parameter: {inputs[2]!r}")
    if list(inputs[3]) != [8, 1024, 3072]:
        raise ValueError(f"unexpected second view shape parameter: {inputs[3]!r}")
    if list(inputs[4]) != [M, N]:
        raise ValueError(f"unexpected flattened view shape parameter: {inputs[4]!r}")
    if list(inputs[5]) != [N]:
        raise ValueError(f"unexpected sum view shape parameter: {inputs[5]!r}")
    return mm, x


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
    mm, x = _validate_inputs(inputs)
    base_out = torch.empty_strided((M, N), (N, 1), device=mm.device, dtype=torch.float32)
    partials = torch.empty_strided((ROW_BLOCKS, N), (N, 1), device=mm.device, dtype=torch.float32)
    sum_out = torch.empty_strided((N,), (1,), device=mm.device, dtype=torch.float32)

    store_grid = lambda meta: (ROW_BLOCKS, triton.cdiv(N, meta["BLOCK_N"]))
    _store_and_partial_sum_kernel[store_grid](
        mm,
        x,
        base_out,
        partials,
        n_cols=N,
        ROW_BLOCK_=ROW_BLOCK,
    )
    _final_sum_kernel[(triton.cdiv(N, FINAL_BLOCK_N),)](
        partials,
        sum_out,
        n_cols=N,
        ROW_BLOCKS_=ROW_BLOCKS,
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=8,
    )
    transposed = torch.as_strided(base_out, (N, M), (1, N))
    return (transposed, sum_out)


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
