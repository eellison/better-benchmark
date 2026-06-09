"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BART logits trim-and-bias scope as one Triton pointwise layout kernel, loading `mm[:, :50265]` with the original 50272-column row stride, broadcasting the `[1, 50265]` bias by column, and writing the required contiguous `[1, 512, 50265]` fp16 output, whereas tuned Inductor already lowers the captured slice/view/add graph to the same single-kernel memory-traffic envelope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a narrower new pattern because the remaining work is the mandatory strided input read, broadcast bias read, and output store; the fix is BANDWIDTH_BOUND: record this row as at floor unless broader pointwise layout or launch-overhead work moves both implementations."""
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

ROWS = 512
INPUT_COLS = 50272
COLS = 50265
INPUT_SHAPE = (ROWS, INPUT_COLS)
BIAS_SHAPE = (1, COLS)
OUT_SHAPE = (1, ROWS, COLS)
OUT_STRIDE = (ROWS * COLS, COLS, 1)
N_ELEMENTS = ROWS * COLS
CLASSIFICATION = "BANDWIDTH_BOUND"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
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

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 512}, num_warps=4, num_stages=4),
            triton.Config({"XBLOCK": 1024}, num_warps=4, num_stages=4),
            triton.Config({"XBLOCK": 2048}, num_warps=8, num_stages=4),
            triton.Config({"XBLOCK": 4096}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _trim_bias_add_kernel(
        mm_ptr,
        bias_ptr,
        out_ptr,
        N: tl.constexpr,
        COLS: tl.constexpr,
        INPUT_COLS: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = offsets < N
        cols = offsets % COLS
        rows = offsets // COLS

        mm_values = tl.load(
            mm_ptr + cols + INPUT_COLS * rows,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        bias_values = tl.load(
            bias_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        tl.store(out_ptr + offsets, mm_values + bias_values, mask=mask)


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    actual = tuple(int(dim) for dim in value)
    if actual != expected:
        raise ValueError(f"{name} is {actual}, expected {expected}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_bias_add.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    mm, arg2_1, shape0 = inputs
    if not isinstance(mm, torch.Tensor) or not isinstance(arg2_1, torch.Tensor):
        raise TypeError("oracle_layout_bias_add.py expects two tensor inputs followed by a shape parameter")
    if mm.device.type != "cuda" or arg2_1.device.type != "cuda":
        raise RuntimeError("oracle_layout_bias_add.py expects CUDA tensor inputs")
    if mm.device != arg2_1.device:
        raise RuntimeError(f"input devices must match, got {mm.device} and {arg2_1.device}")
    if mm.dtype != torch.float16 or arg2_1.dtype != torch.float16:
        raise TypeError(f"expected fp16 inputs, got {mm.dtype} and {arg2_1.dtype}")
    if tuple(mm.shape) != INPUT_SHAPE:
        raise ValueError(f"mm has shape {tuple(mm.shape)}, expected {INPUT_SHAPE}")
    if tuple(arg2_1.shape) != BIAS_SHAPE:
        raise ValueError(f"arg2_1 has shape {tuple(arg2_1.shape)}, expected {BIAS_SHAPE}")
    if tuple(mm.stride()) != (INPUT_COLS, 1):
        raise ValueError(f"mm has stride {tuple(mm.stride())}, expected {(INPUT_COLS, 1)}")
    if tuple(arg2_1.stride()) != (COLS, 1):
        raise ValueError(f"arg2_1 has stride {tuple(arg2_1.stride())}, expected {(COLS, 1)}")
    _require_shape("_shape_param_0", shape0, OUT_SHAPE)
    return mm, arg2_1


@oracle_impl(hardware="H100", shapes="(T([512, 50272], f16), T([1, 50265], f16), S([1, 512, 50265]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    mm, arg2_1 = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=mm.device,
        dtype=mm.dtype,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["XBLOCK"]),)
    _trim_bias_add_kernel[grid](
        mm,
        arg2_1,
        output,
        N=N_ELEMENTS,
        COLS=COLS,
        INPUT_COLS=INPUT_COLS,
    )
    return output


def _check_layout(output: torch.Tensor) -> bool:
    return (
        tuple(output.shape) == OUT_SHAPE
        and tuple(output.stride()) == OUT_STRIDE
        and output.dtype == torch.float16
        and output.storage_offset() == 0
    )


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
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()} "
            f"dtype={layout_out.dtype})"
        )
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
