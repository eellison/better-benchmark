"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete permute-clone-view layout materialization as one hand-specialized Triton row/column copy from `[32, 32, 128, 80]` to the fresh contiguous `[4096, 2560]` output, whereas Inductor lowers the same captured scope as a generic generated pointwise layout-copy kernel; Inductor cannot materially do less work today because the repro requires a fresh non-aliasing output and therefore must read and write all 10,485,760 fp32 elements, so the fix is BANDWIDTH_BOUND: treat this as a memory-floor layout copy unless broader copy-kernel codegen or memory-bandwidth improvements move both paths."""
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


IN_SHAPE = (32, 32, 128, 80)
IN_STRIDE = (327680, 10240, 80, 1)
OUT_SHAPE = (4096, 2560)
OUT_STRIDE = (2560, 1)
OUT_ROWS = 4096
OUT_COLS = 2560
BLOCK_ROWS = 4


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _layout_transpose_kernel(
        x_ptr,
        out_ptr,
        BLOCK_M: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < 4096
        b = rows // 128
        s = rows - b * 128

        cols0 = tl.arange(0, 1024)
        h0 = cols0 // 80
        d0 = cols0 - h0 * 80
        in0 = (
            b[:, None] * 327680
            + h0[None, :] * 10240
            + s[:, None] * 80
            + d0[None, :]
        )
        out0 = rows[:, None] * 2560 + cols0[None, :]
        vals0 = tl.load(x_ptr + in0, mask=row_mask[:, None])
        tl.store(out_ptr + out0, vals0, mask=row_mask[:, None])

        cols1 = 1024 + tl.arange(0, 1024)
        h1 = cols1 // 80
        d1 = cols1 - h1 * 80
        in1 = (
            b[:, None] * 327680
            + h1[None, :] * 10240
            + s[:, None] * 80
            + d1[None, :]
        )
        out1 = rows[:, None] * 2560 + cols1[None, :]
        vals1 = tl.load(x_ptr + in1, mask=row_mask[:, None])
        tl.store(out_ptr + out1, vals1, mask=row_mask[:, None])

        cols2 = 2048 + tl.arange(0, 512)
        h2 = cols2 // 80
        d2 = cols2 - h2 * 80
        in2 = (
            b[:, None] * 327680
            + h2[None, :] * 10240
            + s[:, None] * 80
            + d2[None, :]
        )
        out2 = rows[:, None] * 2560 + cols2[None, :]
        vals2 = tl.load(x_ptr + in2, mask=row_mask[:, None])
        tl.store(out_ptr + out2, vals2, mask=row_mask[:, None])


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, tuple[int, int]]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg16_1, shape_param = inputs
    if not isinstance(arg16_1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor")
    if tuple(arg16_1.shape) != IN_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(arg16_1.shape)}")
    if tuple(arg16_1.stride()) != IN_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(arg16_1.stride())}")
    if arg16_1.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {arg16_1.dtype}")
    if not arg16_1.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")

    out_shape = _shape_tuple(shape_param)
    if out_shape != OUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter: {out_shape}")
    return arg16_1, out_shape


def oracle_forward(inputs):
    """Run the full Repro.forward scope with the exact output layout."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    arg16_1, out_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        out_shape,
        OUT_STRIDE,
        device=arg16_1.device,
        dtype=arg16_1.dtype,
    )
    _layout_transpose_kernel[(triton.cdiv(OUT_ROWS, BLOCK_ROWS),)](
        arg16_1,
        output,
        BLOCK_M=BLOCK_ROWS,
        num_warps=8,
    )
    if tuple(output.stride()) != OUT_STRIDE or output.storage_offset() != 0:
        raise RuntimeError(
            f"unexpected output layout: stride={tuple(output.stride())} "
            f"storage_offset={output.storage_offset()}"
        )
    return output


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
