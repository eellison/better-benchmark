"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Qwen segment-causal attention-bias construction returned by Repro.forward, including the iota-derived causal predicate, both cumsum advanced-index loads, the segment-equality predicate, and all four independent bf16 where outputs in one Triton pass, whereas tuned Inductor's generic mask construction and repeated pointwise bias stores already time within noise of the full-scope oracle for this captured shape; Inductor cannot materially improve this case through a gap-specific scheduler change because the remaining work is one small structured predicate materialization with required output stores; the fix is BANDWIDTH_BOUND: record this as an at-floor segment-mask case unless broader pointwise launch or mask-codegen work moves the baseline."""
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
OUT_NUMEL = BATCH * SEQ * SEQ


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects cumsum and one shape parameter, got {len(inputs)}")
    cumsum, shape_param = inputs
    if not isinstance(cumsum, torch.Tensor):
        raise TypeError(f"expected tensor cumsum input, got {type(cumsum)!r}")
    if tuple(cumsum.shape) != (BATCH, SEQ) or cumsum.dtype != torch.int64:
        raise ValueError(
            f"unexpected cumsum metadata: shape={tuple(cumsum.shape)} dtype={cumsum.dtype}"
        )
    if _shape_tuple(shape_param) != (BATCH, -1, SEQ, SEQ):
        raise ValueError(f"unexpected expand shape parameter {shape_param!r}")
    return cumsum


if triton is not None:

    @triton.jit
    def _segment_causal_bias4_kernel(
        cumsum_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        SEQ_LEN: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // SEQ_LEN
        query = row % SEQ_LEN
        key = tl.arange(0, BLOCK_N)
        active = key < SEQ_LEN

        query_segment = tl.load(cumsum_ptr + batch * SEQ_LEN + query)
        key_segment = tl.load(cumsum_ptr + batch * SEQ_LEN + key, mask=active, other=1)
        keep = (key <= query) & (query_segment == key_segment) & active
        values = tl.where(keep, 0.0, -float("inf"))
        offsets = row * SEQ_LEN + key

        tl.store(out0_ptr + offsets, values, mask=active)
        tl.store(out1_ptr + offsets, values, mask=active)
        tl.store(out2_ptr + offsets, values, mask=active)
        tl.store(out3_ptr + offsets, values, mask=active)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full segment-causal attention-bias construction."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    cumsum = _validate_inputs(inputs)

    out0 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=cumsum.device, dtype=torch.bfloat16)
    out1 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=cumsum.device, dtype=torch.bfloat16)
    out2 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=cumsum.device, dtype=torch.bfloat16)
    out3 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=cumsum.device, dtype=torch.bfloat16)
    grid = (BATCH * SEQ,)
    _segment_causal_bias4_kernel[grid](
        cumsum,
        out0,
        out1,
        out2,
        out3,
        SEQ_LEN=SEQ,
        BLOCK_N=512,
        num_warps=8,
    )
    return (out0, out1, out2, out3)


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
            outputs = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = True
        for index, output in enumerate(outputs):
            this_ok = (
                tuple(output.shape) == OUT_SHAPE
                and tuple(output.stride()) == OUT_STRIDE
                and output.dtype == torch.bfloat16
            )
            layout_ok = layout_ok and this_ok
            print(
                f"  output {index} layout: {'PASS' if this_ok else 'FAIL'} "
                f"(shape={list(output.shape)} stride={output.stride()} dtype={output.dtype})"
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
