"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full `tanh(arg2 * permute(arg0) + arg1)` Repro with a rank-aware Triton broadcast tile that reuses the row operand and 16 column operands while writing the returned contiguous `float32[1000, 16]` layout, whereas Inductor fuses the graph into one flattened pointwise Triton kernel that reloads the broadcasted row and column values per output element and still emits the scalar multiply-by-one identities; Inductor cannot do this today because pointwise codegen linearizes broadcast maps instead of selecting a tiled outer-broadcast pattern with invariant-load hoisting and identity folding; the fix is NEW_PATTERN: add a tiled broadcast-pointwise codegen pattern for small outer products that hoists row/column loads and drops scalar identity operations before emission."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None
    libdevice = None

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
CLASSIFICATION = "NEW_PATTERN"

ARG0_SHAPE = (16, 1)
ARG2_SHAPE = (1000, 1)
ARG1_SHAPE = (16,)
OUT_SHAPE = (1000, 16)
OUT_STRIDE = (16, 1)
OUT_ROWS = 1000
OUT_COLS = 16
BLOCK_M = 8


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _broadcast_mul_add_tanh_kernel(
        arg0_ptr,
        arg2_ptr,
        arg1_ptr,
        out_ptr,
        arg0_stride0: tl.constexpr,
        arg2_stride0: tl.constexpr,
        arg1_stride0: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        block_m: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
        col_offsets = tl.arange(0, 16)
        row_mask = row_offsets < rows

        arg2_vals = tl.load(
            arg2_ptr + row_offsets[:, None] * arg2_stride0,
            mask=row_mask[:, None],
            other=0.0,
        )
        arg0_vals = tl.load(arg0_ptr + col_offsets[None, :] * arg0_stride0)
        arg1_vals = tl.load(arg1_ptr + col_offsets[None, :] * arg1_stride0)
        result = libdevice.tanh(arg2_vals * arg0_vals + arg1_vals)
        out_offsets = row_offsets[:, None] * cols + col_offsets[None, :]
        tl.store(out_ptr + out_offsets, result, mask=row_mask[:, None])


def _require_f32_cuda_tensor(
    value: object,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype mismatch: expected torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError("oracle_layout.py expects CUDA tensor inputs")
    return value


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with one Triton pointwise kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    arg0_1 = _require_f32_cuda_tensor(inputs[0], "arg0_1", ARG0_SHAPE)
    arg2_1 = _require_f32_cuda_tensor(inputs[1], "arg2_1", ARG2_SHAPE)
    arg1_1 = _require_f32_cuda_tensor(inputs[2], "arg1_1", ARG1_SHAPE)
    if arg0_1.device != arg2_1.device or arg0_1.device != arg1_1.device:
        raise ValueError(
            f"input device mismatch: {arg0_1.device}, {arg2_1.device}, {arg1_1.device}"
        )

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    _broadcast_mul_add_tanh_kernel[(triton.cdiv(OUT_ROWS, BLOCK_M),)](
        arg0_1,
        arg2_1,
        arg1_1,
        out,
        arg0_stride0=int(arg0_1.stride(0)),
        arg2_stride0=int(arg2_1.stride(0)),
        arg1_stride0=int(arg1_1.stride(0)),
        rows=OUT_ROWS,
        cols=OUT_COLS,
        block_m=BLOCK_M,
        num_warps=4,
        num_stages=4,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[object] | tuple[object, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        tuple(actual.shape) == tuple(expected.shape)
        and actual.dtype == expected.dtype
        and actual.stride() == expected.stride()
        and actual.storage_offset() == expected.storage_offset()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()} "
        f"dtype={actual.dtype} storage_offset={actual.storage_offset()})"
    )
    return ok


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
