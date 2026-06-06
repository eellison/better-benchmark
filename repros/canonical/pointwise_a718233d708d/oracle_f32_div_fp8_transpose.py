"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured fp32 scalar-divide, `float8_e4m3fn` cast, and metadata transpose by writing the fp8-converted contiguous `[50304,768]` base tensor in one Triton pointwise kernel and returning its `[768,50304]` permuted view, whereas Inductor already lowers this isolated pointwise-convert plus view region to the same one-read/one-write memory-traffic envelope; Inductor cannot materially improve it today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because there are no surrounding producers or consumers to fuse and the transpose is only output metadata; the fix is BANDWIDTH_BOUND: record this repro as a pointwise fp8 bandwidth/launch floor unless broader fp8 pointwise codegen or launch-overhead work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "pointwise_a718233d708d"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

M = 50304
N = 768
NUMEL = M * N
DIVISOR = 0.001953125


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _f32_div_fp8_kernel(
        input_ptr,
        output_base_ptr,
        n_elements: tl.constexpr,
        divisor: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_base_ptr + offsets, x / divisor, mask=mask)


def _validate_inputs(inputs: list[object]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (arg1_1,) = inputs
    if not isinstance(arg1_1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input must be a tensor")
    if arg1_1.device.type != "cuda":
        raise ValueError(f"{REPRO_ID} expects a CUDA tensor input")
    if arg1_1.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects fp32 input, got {arg1_1.dtype}")
    if tuple(arg1_1.shape) != (M, N):
        raise ValueError(f"unexpected input shape: {tuple(arg1_1.shape)}")
    if tuple(arg1_1.stride()) != (N, 1):
        raise ValueError(f"unexpected input stride: {tuple(arg1_1.stride())}")

    return arg1_1


def oracle_forward(inputs):
    """Compute the exact Repro.forward output for fp32 divide, fp8 cast, and permute."""
    arg1_1 = _validate_inputs(inputs)
    base = torch.empty_strided(
        (M, N),
        (N, 1),
        device=arg1_1.device,
        dtype=torch.float8_e4m3fn,
    )
    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_SIZE"]),)
    _f32_div_fp8_kernel[grid](
        arg1_1,
        base,
        n_elements=NUMEL,
        divisor=DIVISOR,
    )
    return base.permute(1, 0)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
