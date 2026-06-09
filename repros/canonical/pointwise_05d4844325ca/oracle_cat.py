"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full `full([3], 0.0) -> cat([arg6_1, full])` Repro.forward scope in one Triton materialization kernel by allocating the fresh contiguous f32[50268] output, copying the f32[50265] input prefix, and storing the three constant-zero tail elements directly, whereas Inductor currently lowers the isolated cat/full graph through its generic cat materialization path for the same dense output; Inductor cannot do this today because cat lowering does not have a segment-aware constant-tail template for fixed small concat suffixes and instead relies on general pointwise/cat scheduling machinery; the fix is NEW_PATTERN: add a guarded cat-with-constant-tail materialization template, while measurements that do not beat compiled Inductor should be reported as not a true floor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

IN_N = 50265
TAIL_N = 3
OUT_N = IN_N + TAIL_N
OUT_SHAPE = (OUT_N,)
OUT_STRIDE = (1,)
DTYPE = torch.float32
CLASSIFICATION = "NEW_PATTERN"


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
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
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
    if value.dtype is not DTYPE:
        raise TypeError(f"unexpected input dtype {value.dtype}, expected {DTYPE}")
    if not value.is_cuda:
        raise ValueError("oracle_cat.py expects CUDA inputs")
    return value


@oracle_impl(hardware="H100", shapes="(T([50265], f32))")
def oracle_forward(inputs):
    """Run the exact Repro.forward scope with one Triton copy/fill kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    arg6_1 = _validate_input(inputs[0])
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg6_1.device,
        dtype=DTYPE,
    )
    _cat_zero_tail_kernel[(triton.cdiv(OUT_N, 512),)](
        arg6_1,
        out,
        input_stride0=arg6_1.stride(0),
        IN_N_=IN_N,
        OUT_N_=OUT_N,
        BLOCK_N=512,
        num_warps=4,
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
