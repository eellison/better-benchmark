"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Demucs channel-slice sum returned by Repro.forward, directly reading `arg0_1[:, 1:5, :, 0:382788]` and writing the contiguous f32 `[8,2,382788]` result with four f32 loads and f32 additions per output element, whereas Inductor already has no material intermediate for the slice views and the remaining work is the required channel reads plus output store; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or a new pattern when the direct slice reduction measures at the same memory-traffic floor; the fix is BANDWIDTH_BOUND: record this as at floor unless broader memory-bandwidth or launch-overhead work moves both implementations."""
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


IN_SHAPE = (8, 5, 2, 426888)
IN_STRIDE = (4268880, 853776, 426888, 1)
OUT_SHAPE = (8, 2, 382788)
OUT_STRIDE = (765576, 382788, 1)

BATCH = 8
PAIR = 2
IN_WIDTH = 426888
OUT_WIDTH = 382788
CHANNEL_STRIDE = 853776
OUT_ROWS = BATCH * PAIR


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
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
        ],
        key=["n_cols"],
    )
    @triton.jit
    def _demucs_channel_slice_sum_kernel(
        x_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < n_cols

        batch = row // 2
        pair_index = row - batch * 2
        base = batch * 4268880 + pair_index * 426888 + cols

        c1 = tl.load(
            x_ptr + base + 853776,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        c2 = tl.load(
            x_ptr + base + 2 * 853776,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        c3 = tl.load(
            x_ptr + base + 3 * 853776,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        c4 = tl.load(
            x_ptr + base + 4 * 853776,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        total = c1 + c2
        total = total + c3
        total = total + c4
        tl.store(out_ptr + row * 382788 + cols, total, mask=mask)


def _require_arg0(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")
    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"arg0_1 must be a tensor, got {type(x)!r}")
    if tuple(x.shape) != IN_SHAPE:
        raise ValueError(f"arg0_1 has shape {tuple(x.shape)}, expected {IN_SHAPE}")
    if x.dtype != torch.float32:
        raise TypeError(f"arg0_1 has dtype {x.dtype}, expected torch.float32")
    if not x.is_cuda:
        raise RuntimeError("arg0_1 must be a CUDA tensor for this Triton oracle")
    if tuple(x.stride()) != IN_STRIDE:
        raise ValueError(f"arg0_1 has stride {tuple(x.stride())}, expected {IN_STRIDE}")
    return x


def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same input tuple as Repro.forward() and returns
    the same single contiguous f32 [8,2,382788] tensor. Numerics are the direct
    f32 slice reads and f32 channel sum; no alternate math paths are used.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_demucs_channel_slice_sum.py")

    x = _require_arg0(inputs)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (OUT_ROWS, triton.cdiv(OUT_WIDTH, meta["BLOCK_N"]))
    _demucs_channel_slice_sum_kernel[grid](
        x,
        out,
        n_cols=OUT_WIDTH,
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
