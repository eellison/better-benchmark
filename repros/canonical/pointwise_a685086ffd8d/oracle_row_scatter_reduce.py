"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle measures AT_FLOOR for the complete Blenderbot masked row-wise `index_put(accumulate=True)` scope by using a full zero-fill plus a row-tiled Triton scatter-add that hoists each token index and `idx == 0` predicate across a hidden-column tile, while Inductor emits a comparable two-kernel full zero-fill plus fused index_put atomic scatter; Inductor cannot materially improve this local repro because the required contiguous `[8008, 2560]` zero write and fp32 indexed atomic accumulation dominate the full-scope work; the fix is BANDWIDTH_BOUND: record this as at-floor unless broader memory-traffic, zero-fill, or scatter-atomic changes move both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "BANDWIDTH_BOUND"
OUTPUT_ROWS = 8008
SEQ = 128
HIDDEN = 2560
BLOCK_D = 256


from oracle_harness import (  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _scatter_rows_kernel(
        values_ptr,
        index_ptr,
        out_ptr,
        n_pos: tl.constexpr,
        hidden: tl.constexpr,
        block_d: tl.constexpr,
    ):
        pos = tl.program_id(0)
        d_block = tl.program_id(1)
        cols = d_block * block_d + tl.arange(0, block_d)
        col_mask = cols < hidden

        row = tl.load(index_ptr + pos)
        value = tl.load(values_ptr + pos * hidden + cols, mask=col_mask, other=0.0)

        # Repro computes where(index == 0, 0.0, value) before index_put.
        tl.atomic_add(
            out_ptr + row * hidden + cols,
            value,
            sem="relaxed",
            mask=col_mask & (row != 0),
        )


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    values, index = inputs
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{REPRO_ID} arg0 must be a tensor")
    if not isinstance(index, torch.Tensor):
        raise TypeError(f"{REPRO_ID} arg1 must be a tensor")
    if values.device.type != "cuda" or index.device.type != "cuda":
        raise RuntimeError(f"{REPRO_ID} oracle expects CUDA tensor inputs")
    if values.dtype != torch.float32:
        raise TypeError(f"expected f32 values, got {values.dtype}")
    if index.dtype != torch.int64:
        raise TypeError(f"expected i64 indices, got {index.dtype}")
    if values.dim() != 3 or index.dim() != 2:
        raise ValueError(f"unexpected ranks: values={values.dim()} index={index.dim()}")

    batch, seq, hidden = values.shape
    if tuple(index.shape) != (batch, seq):
        raise ValueError(
            f"index shape {tuple(index.shape)} does not match values batch/seq {(batch, seq)}"
        )
    if seq != SEQ or hidden != HIDDEN:
        raise ValueError(f"expected [B,{SEQ},{HIDDEN}] values, got {tuple(values.shape)}")
    if values.stride() != (SEQ * HIDDEN, HIDDEN, 1):
        raise ValueError(f"expected contiguous values, got stride {values.stride()}")
    if index.stride() != (SEQ, 1):
        raise ValueError(f"expected contiguous indices, got stride {index.stride()}")
    return values, index


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scatter-reduce scope.

    SCOPE INVARIANT: accepts the same `(arg1_1, arg0_1)` inputs as
    Repro.forward() and returns one contiguous f32 `[8008, 2560]` tensor with
    stride `(2560, 1)`.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_row_scatter_reduce.py")

    values, index = _validate_inputs(inputs)
    batch, seq, hidden = values.shape
    n_pos = batch * seq

    out = torch.empty_strided(
        (OUTPUT_ROWS, hidden),
        (hidden, 1),
        device=values.device,
        dtype=torch.float32,
    )
    out.zero_()

    grid = (n_pos, triton.cdiv(hidden, BLOCK_D))
    _scatter_rows_kernel[grid](
        values,
        index,
        out,
        n_pos=n_pos,
        hidden=hidden,
        block_d=BLOCK_D,
        num_warps=8,
    )
    return out


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in out:
            tensors.extend(_normalize_outputs(item))
        return tensors
    return []


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
    torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.dtype == expected_tensor.dtype
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.storage_offset() == expected_tensor.storage_offset()
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} "
            f"dtype={actual_tensor.dtype})"
        )
        ok = ok and layout_ok
    return ok


# --- CLI entry point ---
def main() -> None:
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
        ok = _check_layout(instance, inputs) and ok
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
