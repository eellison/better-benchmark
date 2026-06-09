"""
Oracle for sum_c89cd4b0ece6

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full three-input permute/view/cat materialization and the dim0 sum in one Triton pass, returning the transposed cat view layout plus the reduced vector, whereas Inductor materializes the cat/transpose side output and reduction through generic scheduled kernels with extra launch/scheduling overhead for this tiny fixed layout; Inductor cannot do this today because its scheduler does not fuse multi-output cat layout materialization with a dependent reduction over the same logical cat buffer; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to recognize cat-as-layout plus reduction consumers and emit a single multi-output kernel for the producer and reduction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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


REPRO_ID = "sum_c89cd4b0ece6"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 64
PER_INPUT_COLS = 768
OUT_COLS = 2304
BLOCK_ROWS = 64
BLOCK_COLS = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _cat_transpose_sum_kernel(
        getitem_46,
        getitem_44,
        getitem_45,
        out_transpose,
        out_sum,
        ROWS_: tl.constexpr,
        PER_INPUT_COLS_: tl.constexpr,
        OUT_COLS_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_start = tl.program_id(0) * BLOCK_N
        rows = tl.arange(0, BLOCK_M)[:, None]
        cols = col_start + tl.arange(0, BLOCK_N)[None, :]
        valid = (rows < ROWS_) & (cols < OUT_COLS_)

        source_cols = cols % PER_INPUT_COLS_
        input_offsets = rows * PER_INPUT_COLS_ + source_cols
        from_44 = cols < PER_INPUT_COLS_
        from_45 = (cols >= PER_INPUT_COLS_) & (cols < (2 * PER_INPUT_COLS_))
        from_46 = cols >= (2 * PER_INPUT_COLS_)

        values = tl.load(getitem_44 + input_offsets, mask=valid & from_44, other=0.0)
        values += tl.load(getitem_45 + input_offsets, mask=valid & from_45, other=0.0)
        values += tl.load(getitem_46 + input_offsets, mask=valid & from_46, other=0.0)

        tl.store(out_transpose + rows * OUT_COLS_ + cols, values, mask=valid)

        summed = tl.sum(values, axis=0)
        flat_cols = col_start + tl.arange(0, BLOCK_N)
        tl.store(out_sum + flat_cols, summed, mask=flat_cols < OUT_COLS_)


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 8:
        raise ValueError(f"expected 8 repro inputs, got {len(inputs)}")

    getitem_46, getitem_44, getitem_45 = inputs[:3]
    for name, tensor in (
        ("getitem_46", getitem_46),
        ("getitem_44", getitem_44),
        ("getitem_45", getitem_45),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA tensors")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} expected dtype torch.float32, got {tensor.dtype}")
        if tuple(tensor.shape) != (1, 12, 64, 64):
            raise ValueError(f"{name} expected shape (1, 12, 64, 64), got {tuple(tensor.shape)}")
        if tensor.stride() != (49152, 64, 768, 1):
            raise ValueError(f"{name} expected stride (49152, 64, 768, 1), got {tensor.stride()}")

    if list(inputs[3]) != [1, 64, 768] or list(inputs[4]) != [1, 64, 768] or list(inputs[5]) != [1, 64, 768]:
        raise ValueError("unexpected per-input view shape parameters")
    if list(inputs[6]) != [64, 2304] or list(inputs[7]) != [2304]:
        raise ValueError("unexpected output view shape parameters")

    return getitem_46, getitem_44, getitem_45


@oracle_impl(hardware="H100", shapes="(T([1, 12, 64, 64], f32, stride=(49152, 64, 768, 1)), T([1, 12, 64, 64], f32, stride=(49152, 64, 768, 1)), T([1, 12, 64, 64], f32, stride=(49152, 64, 768, 1)), S([1, 64, 768]), S([1, 64, 768]), S([1, 64, 768]), S([64, 2304]), S([2304]))")
def oracle_forward(inputs):
    """Compute the exact full Repro.forward scope with a fused Triton kernel."""
    getitem_46, getitem_44, getitem_45 = _validate_inputs(inputs)

    out_transpose = torch.empty_strided(
        (OUT_COLS, ROWS),
        (1, OUT_COLS),
        device=getitem_46.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty_strided(
        (OUT_COLS,),
        (1,),
        device=getitem_46.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(OUT_COLS, BLOCK_COLS),)
    _cat_transpose_sum_kernel[grid](
        getitem_46,
        getitem_44,
        getitem_45,
        out_transpose,
        out_sum,
        ROWS_=ROWS,
        PER_INPUT_COLS_=PER_INPUT_COLS,
        OUT_COLS_=OUT_COLS,
        BLOCK_M=BLOCK_ROWS,
        BLOCK_N=BLOCK_COLS,
        num_warps=4,
    )
    return (out_transpose, out_sum)


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
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Disable auto-detection and skipping of stochastic outputs")
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
