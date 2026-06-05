"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Moondream causal same-segment attention-mask scope as one Triton pointwise kernel, including the cumsum equality predicate, causal key<=query predicate, fp16 0/-inf where epilogue, and all 24 distinct contiguous fp16[1,1,512,512] return tensors, whereas Inductor's generic lowering for the decomposed iota/unsqueeze/index/equality/where sibling graph already times within noise of this direct full-scope kernel; Inductor cannot materially improve this local region because the dominant cost is mandatory allocation plus stores for 24 materialized fp16 mask outputs rather than avoidable producer work; the fix is BANDWIDTH_BOUND: record this as at floor unless a larger graph can eliminate or alias the sibling outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


ROWS = 512
COLS = 512
NUM_OUTPUTS = 24
OUT_SHAPE = (1, 1, ROWS, COLS)
OUT_STRIDE = (ROWS * COLS, ROWS * COLS, COLS, 1)
NUMEL = ROWS * COLS
BLOCK_SIZE = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _segment_mask_multi_output_kernel(
        cumsum_ptr,
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
        n_cols: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        query_pos = offsets // n_cols
        key_pos = offsets - query_pos * n_cols

        query_segment = tl.load(cumsum_ptr + query_pos, mask=mask, other=-1)
        key_segment = tl.load(cumsum_ptr + key_pos, mask=mask, other=-2)
        keep = (key_pos <= query_pos) & (query_segment == key_segment) & mask
        values = tl.where(keep, 0.0, -float("inf"))

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


def oracle_forward(inputs):
    """Run the full Repro.forward scope with Triton compute and 24 outputs."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    cumsum, shape_param = inputs
    if tuple(cumsum.shape) != (1, ROWS) or cumsum.dtype != torch.int64:
        raise ValueError(f"expected cumsum i64[1,{ROWS}], got shape={tuple(cumsum.shape)} dtype={cumsum.dtype}")
    if list(shape_param) != [1, -1, ROWS, COLS]:
        raise ValueError(f"unexpected expand shape parameter: {shape_param}")

    outputs = tuple(torch.empty(OUT_SHAPE, device=cumsum.device, dtype=torch.float16) for _ in range(NUM_OUTPUTS))
    grid = (triton.cdiv(NUMEL, BLOCK_SIZE),)
    _segment_mask_multi_output_kernel[grid](
        cumsum,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
        outputs[8],
        outputs[9],
        outputs[10],
        outputs[11],
        outputs[12],
        outputs[13],
        outputs[14],
        outputs[15],
        outputs[16],
        outputs[17],
        outputs[18],
        outputs[19],
        outputs[20],
        outputs[21],
        outputs[22],
        outputs[23],
        n_cols=COLS,
        n_elements=NUMEL,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
    )
    return outputs


def _normalize_outputs(out):
    if isinstance(out, torch.Tensor):
        return [out]
    result = []
    for item in out:
        if isinstance(item, torch.Tensor):
            result.append(item)
    return result


def _check_layout_and_storage(instance, inputs) -> bool:
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        oracle_out = _normalize_outputs(oracle_forward(inputs))
        torch.cuda.synchronize()

    ok = True
    eager_ptrs = {out.data_ptr() for out in eager}
    oracle_ptrs = {out.data_ptr() for out in oracle_out}
    for idx, (expected, actual) in enumerate(zip(eager, oracle_out)):
        item_ok = (
            tuple(actual.shape) == tuple(expected.shape) == OUT_SHAPE
            and tuple(actual.stride()) == tuple(expected.stride()) == OUT_STRIDE
            and actual.dtype == expected.dtype == torch.float16
            and actual.is_contiguous()
        )
        ok = ok and item_ok
        print(
            f"  output {idx}: {'PASS' if item_ok else 'FAIL'} layout "
            f"(shape={list(actual.shape)} stride={tuple(actual.stride())})"
        )

    storage_ok = len(eager_ptrs) == len(oracle_ptrs) == NUM_OUTPUTS
    print(
        f"  output storage: {'PASS' if storage_ok else 'FAIL'} "
        f"(eager_unique={len(eager_ptrs)} oracle_unique={len(oracle_ptrs)})"
    )
    return ok and storage_ok


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
        ok = _check_layout_and_storage(instance, inputs) and ok
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
