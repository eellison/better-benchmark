"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full repeated BART-style causal attention-mask repro as one shape-specialized Triton fill over twelve distinct f32[8,1,1024,1024] bases and returns the exact f32[8,16,1024,1024] stride-zero expanded views, whereas Inductor lowers each repeated iota/le/where/expand chain as generic pointwise mask materialization work; Inductor cannot do this today because it has no canonical causal-mask pattern that recognizes the repeated metadata-only head expansion while preserving distinct output storages; the fix is NEW_PATTERN: add a causal attention-mask lowering that emits one fused multi-output fill kernel and returns the required expanded view metadata."""
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

BATCH = 8
HEADS = 16
ROWS = 1024
COLS = 1024
NUM_OUTPUTS = 12
BASE_SHAPE = (BATCH, 1, ROWS, COLS)
BASE_STRIDE = (ROWS * COLS, ROWS * COLS, COLS, 1)
EXPANDED_SHAPE = (BATCH, HEADS, ROWS, COLS)
EXPANDED_STRIDE = (ROWS * COLS, 0, COLS, 1)
BASE_NUMEL = BATCH * ROWS * COLS
BLOCK_SIZE = 1024


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _causal_mask_12_kernel(
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
        n_elements: tl.constexpr,
        n_cols: tl.constexpr,
        n_rows: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < n_elements
        row_index = (offsets // n_cols) % n_rows
        col_index = offsets % n_cols
        keep = col_index <= row_index
        values = tl.where(keep, 0.0, -float("inf"))

        tl.store(out0 + offsets, values, mask=valid)
        tl.store(out1 + offsets, values, mask=valid)
        tl.store(out2 + offsets, values, mask=valid)
        tl.store(out3 + offsets, values, mask=valid)
        tl.store(out4 + offsets, values, mask=valid)
        tl.store(out5 + offsets, values, mask=valid)
        tl.store(out6 + offsets, values, mask=valid)
        tl.store(out7 + offsets, values, mask=valid)
        tl.store(out8 + offsets, values, mask=valid)
        tl.store(out9 + offsets, values, mask=valid)
        tl.store(out10 + offsets, values, mask=valid)
        tl.store(out11 + offsets, values, mask=valid)


def _shape_tuple(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)  # type: ignore[arg-type]


def _validate_inputs(inputs: list[object] | tuple[object, ...]) -> None:
    if len(inputs) != NUM_OUTPUTS + 1:
        raise ValueError(f"{REPRO_ID} expects {NUM_OUTPUTS + 1} shape inputs, got {len(inputs)}")
    if _shape_tuple(inputs[0]) != (BATCH, -1, ROWS, COLS):
        raise ValueError(f"unexpected first expand shape: {inputs[0]}")
    for index, shape in enumerate(inputs[1:], start=1):
        if _shape_tuple(shape) != EXPANDED_SHAPE:
            raise ValueError(f"unexpected output shape input {index}: {shape}")


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope with a fused causal-mask fill kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    _validate_inputs(inputs)

    bases = [
        torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device="cuda", dtype=torch.float32)
        for _ in range(NUM_OUTPUTS)
    ]
    grid = (triton.cdiv(BASE_NUMEL, BLOCK_SIZE),)
    _causal_mask_12_kernel[grid](
        bases[0],
        bases[1],
        bases[2],
        bases[3],
        bases[4],
        bases[5],
        bases[6],
        bases[7],
        bases[8],
        bases[9],
        bases[10],
        bases[11],
        n_elements=BASE_NUMEL,
        n_cols=COLS,
        n_rows=ROWS,
        BLOCK=BLOCK_SIZE,
        num_warps=8,
    )
    return tuple(base.expand(EXPANDED_SHAPE) for base in bases)


def _normalize_outputs(out: object) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [item for item in out if isinstance(item, torch.Tensor)]
    return []


def _check_layout_and_aliases(instance: torch.nn.Module, inputs: list[object]) -> bool:
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        actual = _normalize_outputs(oracle_forward(inputs))
        torch.cuda.synchronize()

    if len(eager) != NUM_OUTPUTS or len(actual) != NUM_OUTPUTS:
        print(f"  layout: FAIL output_count eager={len(eager)} oracle={len(actual)}")
        return False

    ok = True
    eager_storages = [
        (tensor.untyped_storage().data_ptr(), tensor.storage_offset()) for tensor in eager
    ]
    actual_storages = [
        (tensor.untyped_storage().data_ptr(), tensor.storage_offset()) for tensor in actual
    ]

    for index, (expected, observed) in enumerate(zip(eager, actual)):
        item_ok = (
            tuple(observed.shape) == tuple(expected.shape) == EXPANDED_SHAPE
            and observed.dtype == expected.dtype == torch.float32
            and tuple(observed.stride()) == tuple(expected.stride()) == EXPANDED_STRIDE
            and observed.storage_offset() == expected.storage_offset() == 0
            and observed.untyped_storage().nbytes() == expected.untyped_storage().nbytes()
        )
        ok = ok and item_ok
        print(
            f"  output {index} layout: {'PASS' if item_ok else 'FAIL'} "
            f"(stride={tuple(observed.stride())} storage_offset={observed.storage_offset()} "
            f"storage_nbytes={observed.untyped_storage().nbytes()})"
        )

    alias_ok = True
    for left in range(NUM_OUTPUTS):
        for right in range(left + 1, NUM_OUTPUTS):
            expected_alias = eager_storages[left] == eager_storages[right]
            observed_alias = actual_storages[left] == actual_storages[right]
            alias_ok = alias_ok and expected_alias == observed_alias

    print(
        f"  inter-output aliases: {'PASS' if alias_ok else 'FAIL'} "
        f"(distinct_storages={len({storage for storage, _ in actual_storages})})"
    )
    return ok and alias_ok


def main() -> None:
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
        ok = _check_layout_and_aliases(instance, inputs) and ok
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
