"""
Full-scope oracle for pointwise_b43349159514.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle materializes the exact
`cat([arg6_1, full([7], 0, f16)])` result with one Triton kernel that copies the
contiguous f16[50265] input segment and fills the seven-element zero tail
directly into the returned contiguous f16[50272] layout. The tuned Inductor
harness compile path already beats this timed copy/fill artifact, and the
required interleaved compile configs are in the same one-launch materialization
range for about 200 KiB of input/output traffic. Inductor cannot materially do
less work without removing the required fresh output allocation, launch, or
copy/store traffic; the fix is BANDWIDTH_BOUND: no scheduler, scatter-reduce,
split-K, algebraic, recompute, or new-pattern compiler change is indicated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - py_compile should not require Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

IN_N = 50265
TAIL_N = 7
OUT_N = IN_N + TAIL_N
OUT_SHAPE = (OUT_N,)
OUT_STRIDE = (1,)
CLASSIFICATION = "BANDWIDTH_BOUND"


from oracle_harness import (  # noqa: E402
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


if triton is not None:

    @triton.jit
    def _cat_zero_tail_kernel(
        input_ptr,
        output_ptr,
        input_stride0: tl.constexpr,
        IN_N_: tl.constexpr,
        OUT_N_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        output_mask = offsets < OUT_N_
        copy_mask = offsets < IN_N_
        values = tl.load(
            input_ptr + offsets * input_stride0,
            mask=copy_mask,
            other=0.0,
        )
        tl.store(output_ptr + offsets, values, mask=output_mask)


def _validate_input(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor input, got {type(value)!r}")
    if tuple(value.shape) != (IN_N,):
        raise ValueError(f"unexpected input shape {tuple(value.shape)}, expected ({IN_N},)")
    if value.dtype is not torch.float16:
        raise TypeError(f"unexpected input dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise ValueError("oracle_layout.py expects CUDA inputs")
    return value


def oracle_forward(inputs):
    """Run the full Repro.forward scope with one Triton copy/fill kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    arg6_1 = _validate_input(inputs[0])
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg6_1.device,
        dtype=torch.float16,
    )
    _cat_zero_tail_kernel[(triton.cdiv(OUT_N, 2048),)](
        arg6_1,
        out,
        input_stride0=arg6_1.stride(0),
        IN_N_=IN_N,
        OUT_N_=OUT_N,
        BLOCK_N=2048,
        num_warps=16,
    )
    return out


def _check_layout(output: torch.Tensor) -> bool:
    return tuple(output.shape) == OUT_SHAPE and output.stride() == OUT_STRIDE


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
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()})"
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
