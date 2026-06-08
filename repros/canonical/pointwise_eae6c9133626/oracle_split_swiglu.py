"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Qwen split-SwiGLU pointwise scope with a row/column Triton kernel that reads the two halves of the captured contiguous bf16 `[16384, 1536]` input, performs the fp32 `x / (libdevice.exp(-x) + 1)` activation used by Inductor, preserves the eager bf16 rounding boundary before multiplying by the gate half, and writes the required contiguous `[16384, 768]` output directly, whereas Inductor emits one generic flattened pointwise kernel that recovers split-view row and column indices with integer div/mod; Inductor cannot do this today because the pointwise scheduler/codegen does not recognize this split-last-dimension gated activation as a dedicated layout-aware pattern with affine half-row loads and the required low-precision boundary; the fix is NEW_PATTERN: add a guarded split-SwiGLU pointwise template, or teach generic pointwise codegen to specialize split-last-dim indexing into the same row/column affine addressing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
    get_shape_key,
    has_stochastic_ops,
)


M = 16384
N = 768
INPUT_SHAPE = (M, 2 * N)
OUTPUT_SHAPE = (M, N)
INPUT_STRIDE = (2 * N, 1)
OUTPUT_STRIDE = (N, 1)


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
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        ],
        key=["M_SIZE", "N_SIZE"],
    )
    @triton.jit
    def _split_swiglu_kernel(
        x_ptr,
        out_ptr,
        M_SIZE: tl.constexpr,
        N_SIZE: tl.constexpr,
        INPUT_ROW_STRIDE: tl.constexpr,
        OUTPUT_ROW_STRIDE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M_SIZE) & (cols[None, :] < N_SIZE)

        lhs_offsets = rows[:, None] * INPUT_ROW_STRIDE + cols[None, :]
        rhs_offsets = lhs_offsets + N_SIZE
        out_offsets = rows[:, None] * OUTPUT_ROW_STRIDE + cols[None, :]

        lhs = tl.load(x_ptr + lhs_offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(x_ptr + rhs_offsets, mask=mask, other=0.0).to(tl.float32)

        activated = lhs / (libdevice.exp(-lhs) + 1.0)
        activated_bf16 = activated.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + out_offsets, activated_bf16 * rhs, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_split_swiglu.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (x,) = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"input must be a tensor, got {type(x)!r}")
    if x.device.type != "cuda":
        raise RuntimeError("oracle_split_swiglu.py expects CUDA inputs")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {x.dtype}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != INPUT_STRIDE or x.storage_offset() != 0:
        raise ValueError(
            f"expected captured contiguous input layout, got stride={x.stride()} "
            f"storage_offset={x.storage_offset()}"
        )
    return x


def oracle_forward(inputs):
    """Run the full repro computation."""
    x = _validate_inputs(inputs)
    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _split_swiglu_kernel[grid](
        x,
        out,
        M_SIZE=M,
        N_SIZE=N,
        INPUT_ROW_STRIDE=INPUT_STRIDE[0],
        OUTPUT_ROW_STRIDE=OUTPUT_STRIDE[0],
    )
    return out


def _check_layout_and_alias(instance, inputs) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        isinstance(eager, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(eager.shape)
        and actual.dtype == eager.dtype
        and actual.device == eager.device
        and tuple(actual.stride()) == tuple(eager.stride())
        and actual.storage_offset() == eager.storage_offset()
        and (actual._base is None) == (eager._base is None)
        and actual.data_ptr() != inputs[0].data_ptr()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()} "
        f"storage_offset={actual.storage_offset()} base_is_none={actual._base is None})"
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
        ok = ok and _check_layout_and_alias(instance, inputs)
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
