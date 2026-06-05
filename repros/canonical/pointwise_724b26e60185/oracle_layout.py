"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the full `full([2], 0.0) -> cat([arg204_1, full])` repro by copying the contiguous `float32[30522]` prefix and writing the two zero tail elements in one dedicated cat-with-constant-tail Triton kernel, whereas Inductor currently emits a generic pointwise fused_cat_full kernel over the concatenated index space with per-element range predicates, a masked input load, `tl.where` source muxing, and constant-zero fallback; Inductor cannot do this today because cat lowering goes through generic pointwise index codegen and has no segment-aware template for fixed concat of one contiguous tensor with a tiny constant full tail; the fix is NEW_PATTERN: add a fixed-segment cat/full materialization template that emits direct contiguous prefix copies plus constant tail stores without per-element source mux logic."""
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

IN_NUMEL = 30522
OUT_NUMEL = 30524
OUT_SHAPE = (OUT_NUMEL,)
OUT_STRIDE = (1,)
DTYPE = torch.float32
BLOCK_N = 512
NUM_WARPS = 4
CLASSIFICATION = "NEW_PATTERN"


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
    def _cat_zero_kernel(
        in_ptr,
        out_ptr,
        input_stride0: tl.constexpr,
        OUT_N: tl.constexpr,
        IN_N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        out_mask = offsets < OUT_N
        input_mask = offsets < IN_N
        values = tl.load(in_ptr + offsets * input_stride0, mask=input_mask, other=0.0)
        tl.store(out_ptr + offsets, values, mask=out_mask)


def oracle_forward(inputs):
    """Run the exact Repro.forward scope with one Triton materialization kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input tensor, got {len(inputs)}")

    (arg204_1,) = inputs
    if not isinstance(arg204_1, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(arg204_1)!r}")
    if tuple(arg204_1.shape) != (IN_NUMEL,):
        raise ValueError(f"expected f32[{IN_NUMEL}], got shape={tuple(arg204_1.shape)}")
    if arg204_1.dtype is not DTYPE:
        raise ValueError(f"expected dtype {DTYPE}, got {arg204_1.dtype}")
    if not arg204_1.is_cuda:
        raise ValueError("oracle_layout.py expects CUDA inputs")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg204_1.device,
        dtype=DTYPE,
    )
    grid = (triton.cdiv(OUT_NUMEL, BLOCK_N),)
    _cat_zero_kernel[grid](
        arg204_1,
        out,
        input_stride0=arg204_1.stride(0),
        OUT_N=OUT_NUMEL,
        IN_N=IN_NUMEL,
        BLOCK=BLOCK_N,
        num_warps=NUM_WARPS,
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
