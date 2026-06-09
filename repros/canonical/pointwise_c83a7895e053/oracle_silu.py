"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete dense f32 SiLU pointwise scope `x / (exp(-x) + 1)` in one flat Triton activation kernel while preserving the eager output shape and dense input layout, whereas Inductor lowers the same decomposed neg/exp/add/div graph as a generic pointwise kernel that keeps the aten exp as a libdevice call inside the generic scheduler; Inductor cannot do this today because its pointwise pattern/codegen stack does not recognize this dense SiLU idiom as a benchmark-gated specialized activation lowering with the lean Triton math path; the fix is NEW_PATTERN: add a guarded dense SiLU pointwise lowering, or make generic pointwise codegen select the same fast expression and launch shape when correctness tolerances allow it."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _silu_nomask_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(input_ptr + offsets).to(tl.float32)
        out = x / (tl.exp(-x) + 1.0)
        tl.store(output_ptr + offsets, out)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _silu_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = x / (tl.exp(-x) + 1.0)
        tl.store(output_ptr + offsets, out, mask=mask)


def _dense_storage_size(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    if len(shape) != len(stride):
        raise ValueError("shape and stride rank mismatch")
    if any(dim < 0 for dim in shape):
        raise ValueError(f"invalid negative shape: {shape}")
    if any(dim == 0 for dim in shape):
        return 0
    return 1 + sum((dim - 1) * step for dim, step in zip(shape, stride) if dim > 1)


def _validate_input(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_silu.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (x,) = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input must be a tensor, got {type(x)!r}")
    if x.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects f32 input, got {x.dtype}")
    if not x.is_cuda:
        raise RuntimeError(f"{REPRO_ID} expects a CUDA input")

    shape = tuple(x.shape)
    stride = tuple(x.stride())
    if x.storage_offset() != 0:
        raise ValueError(f"{REPRO_ID} expects zero storage_offset, got {x.storage_offset()}")
    if _dense_storage_size(shape, stride) != x.numel():
        raise ValueError(
            f"{REPRO_ID} expects a dense input layout, got shape={shape} stride={stride}"
        )
    return x


@oracle_impl(hardware="H100", shapes="(T([128, 384, 7, 7], f32))")
def oracle_forward(inputs):
    """Run the full-scope SiLU pointwise computation."""
    x = _validate_input(inputs)
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )
    if x.numel() % 2048 == 0:
        grid = lambda meta: (x.numel() // meta["BLOCK_SIZE"],)
        _silu_nomask_kernel[grid](
            x,
            output,
            N=x.numel(),
        )
    else:
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        _silu_kernel[grid](
            x,
            output,
            N=x.numel(),
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
