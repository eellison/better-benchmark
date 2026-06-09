"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete causal segment attention-mask repro as one shape-specialized Triton pointwise kernel over the compact f32[1,1,128,128] base and returns all 28 f32[1,16,128,128] stride-zero expanded aliases, whereas Inductor lowers the repeated iota cumsum-equality causal-mask and expand returns as generic pointwise/view scheduling around the decomposed graph; Inductor cannot do this today because its pattern library does not canonicalize this causal segment-mask construction while preserving shared expanded alias outputs for the complete return tuple; the fix is NEW_PATTERN: add a segment attention-mask lowering that emits one fused mask-fill kernel and returns all sibling expanded aliases from the same base storage."""
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


ROWS = 128
COLS = 128
NUM_OUTPUTS = 28
BASE_SHAPE = (1, 1, ROWS, COLS)
EXPANDED_SHAPE = (1, 16, ROWS, COLS)
EXPANDED_STRIDE = (ROWS * COLS, 0, COLS, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _attention_mask_kernel(
        cumsum_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements
        query_pos = offsets // n_cols
        key_pos = offsets - query_pos * n_cols

        query_segment = tl.load(cumsum_ptr + query_pos, mask=valid, other=-1)
        key_segment = tl.load(cumsum_ptr + key_pos, mask=valid, other=-2)
        keep = (key_pos <= query_pos) & (key_segment == query_segment) & valid
        values = tl.where(keep, 0.0, -3.4028234663852886e38)
        tl.store(out_ptr + offsets, values, mask=valid)


@oracle_impl(hardware="H100", shapes="(T([1, 128], i64, gen=Index(1)), S([1, -1, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]), S([1, 16, 128, 128]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with Triton compute and view outputs."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    cumsum = inputs[0]
    if tuple(cumsum.shape) != (1, ROWS) or cumsum.dtype != torch.int64:
        raise ValueError(f"expected cumsum i64[1,{ROWS}], got shape={tuple(cumsum.shape)} dtype={cumsum.dtype}")

    base = torch.empty(BASE_SHAPE, device=cumsum.device, dtype=torch.float32)
    grid = (triton.cdiv(ROWS * COLS, 256),)
    _attention_mask_kernel[grid](
        cumsum,
        base,
        n_cols=COLS,
        n_elements=ROWS * COLS,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    expanded = base.expand(EXPANDED_SHAPE)
    return (expanded,) * NUM_OUTPUTS


def _normalize_outputs(out):
    if isinstance(out, torch.Tensor):
        return [out]
    result = []
    for item in out:
        if isinstance(item, torch.Tensor):
            result.append(item)
    return result


def _check_layout_and_alias(instance, inputs) -> bool:
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        actual = _normalize_outputs(oracle_forward(inputs))

    ok = True
    if len(actual) != NUM_OUTPUTS or len(eager) != NUM_OUTPUTS:
        print(f"  layout: FAIL output_count eager={len(eager)} oracle={len(actual)}")
        return False

    ref_ptr = actual[0].data_ptr()
    eager_ref_ptr = eager[0].data_ptr()
    for idx, (expected, got) in enumerate(zip(eager, actual)):
        stride_ok = tuple(got.stride()) == tuple(expected.stride()) == EXPANDED_STRIDE
        alias_ok = got.data_ptr() == ref_ptr
        eager_alias_ok = expected.data_ptr() == eager_ref_ptr
        item_ok = stride_ok and alias_ok and eager_alias_ok
        ok = ok and item_ok
        status = "PASS" if item_ok else "FAIL"
        print(
            f"  output {idx}: {status} layout "
            f"(stride={tuple(got.stride())} ref_stride={tuple(expected.stride())} "
            f"oracle_alias={alias_ok} eager_alias={eager_alias_ok})"
        )
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
        layout_ok = _check_layout_and_alias(instance, inputs)
        ok = ok and layout_ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
