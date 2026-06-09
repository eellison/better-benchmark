"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full no-input LLaVA causal-mask construction by materializing all thirty-two distinct fp16 [1,1,512,512] outputs in one Triton multi-output pointwise kernel that shares the iota/le predicate, whereas Inductor currently schedules the thirty-two identical where outputs as thirty-two separate pointwise kernels; Inductor cannot do this today because the pointwise scheduler does not group same-domain graph-output siblings with shared index-derived producers into one multi-output store kernel; the fix is SCHEDULER_FUSION: add graph-output sibling fusion for identical iteration-domain pointwise roots so one schedule can emit all required output stores while reusing the structured predicate."""
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
    oracle_impl,
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

SEQ = 512
OUT_SHAPE = (1, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
OUT_NUMEL = SEQ * SEQ
OUTPUT_COUNT = 32
BLOCK_SIZE = 256
OUT_DEVICE = torch.device("cuda", 0)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> None:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 shape input, got {len(inputs)}")

    expand_shape = _shape_tuple(inputs[0])
    expected = (1, -1, SEQ, SEQ)
    if expand_shape != expected:
        raise ValueError(f"unexpected expand shape {expand_shape}, expected {expected}")


if triton is not None:

    @triton.jit
    def _causal_mask_32_kernel(
        out0,
        out1,
        out2,
        out3,
        out4,
        out5,
        out6,
        out7,
        out8,
        out9,
        out10,
        out11,
        out12,
        out13,
        out14,
        out15,
        out16,
        out17,
        out18,
        out19,
        out20,
        out21,
        out22,
        out23,
        out24,
        out25,
        out26,
        out27,
        out28,
        out29,
        out30,
        out31,
        S: tl.constexpr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        rows = offsets // S
        cols = offsets - rows * S
        values = tl.where(cols <= rows, 0.0, -float("inf"))

        tl.store(out0 + offsets, values, mask=mask)
        tl.store(out1 + offsets, values, mask=mask)
        tl.store(out2 + offsets, values, mask=mask)
        tl.store(out3 + offsets, values, mask=mask)
        tl.store(out4 + offsets, values, mask=mask)
        tl.store(out5 + offsets, values, mask=mask)
        tl.store(out6 + offsets, values, mask=mask)
        tl.store(out7 + offsets, values, mask=mask)
        tl.store(out8 + offsets, values, mask=mask)
        tl.store(out9 + offsets, values, mask=mask)
        tl.store(out10 + offsets, values, mask=mask)
        tl.store(out11 + offsets, values, mask=mask)
        tl.store(out12 + offsets, values, mask=mask)
        tl.store(out13 + offsets, values, mask=mask)
        tl.store(out14 + offsets, values, mask=mask)
        tl.store(out15 + offsets, values, mask=mask)
        tl.store(out16 + offsets, values, mask=mask)
        tl.store(out17 + offsets, values, mask=mask)
        tl.store(out18 + offsets, values, mask=mask)
        tl.store(out19 + offsets, values, mask=mask)
        tl.store(out20 + offsets, values, mask=mask)
        tl.store(out21 + offsets, values, mask=mask)
        tl.store(out22 + offsets, values, mask=mask)
        tl.store(out23 + offsets, values, mask=mask)
        tl.store(out24 + offsets, values, mask=mask)
        tl.store(out25 + offsets, values, mask=mask)
        tl.store(out26 + offsets, values, mask=mask)
        tl.store(out27 + offsets, values, mask=mask)
        tl.store(out28 + offsets, values, mask=mask)
        tl.store(out29 + offsets, values, mask=mask)
        tl.store(out30 + offsets, values, mask=mask)
        tl.store(out31 + offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="(S([1, -1, 512, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full 32-output causal-mask repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    _validate_inputs(inputs)

    outputs = tuple(
        torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=OUT_DEVICE,
            dtype=torch.float16,
        )
        for _ in range(OUTPUT_COUNT)
    )
    _causal_mask_32_kernel[(triton.cdiv(OUT_NUMEL, BLOCK_SIZE),)](
        *outputs,
        S=SEQ,
        N=OUT_NUMEL,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
    )
    return outputs


def _check_layout(outputs: tuple[torch.Tensor, ...]) -> bool:
    if len(outputs) != OUTPUT_COUNT:
        return False
    expected_ptrs = set()
    for out in outputs:
        if (
            tuple(out.shape) != OUT_SHAPE
            or out.stride() != OUT_STRIDE
            or out.dtype != torch.float16
            or not out.is_cuda
        ):
            return False
        expected_ptrs.add(out.data_ptr())
    return len(expected_ptrs) == OUTPUT_COUNT


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
            layout_outputs = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_outputs)
        print(
            f"  output layout/alias: {'PASS' if layout_ok else 'FAIL'} "
            f"(count={len(layout_outputs)} shape={list(OUT_SHAPE)} stride={OUT_STRIDE})"
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
