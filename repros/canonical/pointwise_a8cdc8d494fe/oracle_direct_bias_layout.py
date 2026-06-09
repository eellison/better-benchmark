"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BEiT relative-position-bias gather, head-major padded backing storage, slice, and batch expand by writing gathered values directly into the returned view's storage layout, whereas Inductor reaches the same effective floor for this captured scope; Inductor cannot produce a meaningful targeted speedup today because both implementations must perform one indirect table lookup and one exposed output store per bias element while the zero-stride batch expand already avoids batch materialization; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader memory-layout or indirect-gather codegen improvements move both paths."""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

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


SEQ = 197
HEADS = 12
PADDED_SEQ = 200
BATCH = 128
ELEMENTS = HEADS * SEQ * SEQ
BASE_SHAPE = (1, HEADS, SEQ, PADDED_SEQ)
BASE_STRIDE = (HEADS * SEQ * PADDED_SEQ, SEQ * PADDED_SEQ, PADDED_SEQ, 1)
OUT_SHAPE = (BATCH, HEADS, SEQ, SEQ)
OUT_STRIDE = (0, SEQ * PADDED_SEQ, PADDED_SEQ, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relative_bias_layout_kernel(
        index_ptr,
        table_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        SEQ_N: tl.constexpr,
        HEADS_N: tl.constexpr,
        PADDED_SEQ_N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL

        hw = SEQ_N * SEQ_N
        head = offsets // hw
        pos = offsets - head * hw
        row = pos // SEQ_N
        col = pos - row * SEQ_N

        index = tl.load(index_ptr + row * SEQ_N + col, mask=mask, other=0)
        value = tl.load(table_ptr + index * HEADS_N + head, mask=mask, other=0.0)
        tl.store(
            out_ptr + head * (SEQ_N * PADDED_SEQ_N) + row * PADDED_SEQ_N + col,
            value,
            mask=mask,
        )


def _check_static_scope(index, table, shape0, shape1):
    if tuple(index.shape) != (SEQ, SEQ):
        raise ValueError(f"expected index shape {(SEQ, SEQ)}, got {tuple(index.shape)}")
    if tuple(table.shape) != (732, HEADS):
        raise ValueError(f"expected table shape {(732, HEADS)}, got {tuple(table.shape)}")
    if tuple(shape0) not in ((SEQ, SEQ, HEADS), (SEQ, SEQ, -1)):
        raise ValueError(f"expected reshape shape {(SEQ, SEQ, -1)}, got {tuple(shape0)}")
    if tuple(shape1) != OUT_SHAPE:
        raise ValueError(f"expected expand shape {OUT_SHAPE}, got {tuple(shape1)}")


@oracle_impl(hardware="H100", shapes="(T([197, 197], i64, gen=Index(732)), T([732, 12], f32), S([197, 197, -1]), S([128, 12, 197, 197]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for pointwise_a8cdc8d494fe."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_direct_bias_layout.py")

    index, table, shape0, shape1 = inputs
    _check_static_scope(index, table, shape0, shape1)

    base = torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device=table.device, dtype=table.dtype)
    grid = (triton.cdiv(ELEMENTS, 256),)
    _relative_bias_layout_kernel[grid](
        index,
        table,
        base,
        TOTAL=ELEMENTS,
        SEQ_N=SEQ,
        HEADS_N=HEADS,
        PADDED_SEQ_N=PADDED_SEQ,
        BLOCK=256,
        num_warps=4,
    )
    return torch.as_strided(base, OUT_SHAPE, OUT_STRIDE)


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
