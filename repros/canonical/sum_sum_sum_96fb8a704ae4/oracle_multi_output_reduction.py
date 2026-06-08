"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full MobileBERT captured scope, including the shared `mm_716 + mm_718` producer, the `arg589_1` and `arg8_1` elementwise consumers, the required materialized transposed product output, and all three column reductions in one coalesced row-tiled producer pass with three atomic accumulators, whereas Inductor emits generic reduction/materialization code for the same producer; Inductor cannot do this today because its scheduler does not represent this multi-output reduction plus sibling materialized-view consumer as one shared tiled plan; the fix is SCHEDULER_FUSION: teach the scheduler to form a shared partial-reduction plan for the required output materialization and reduction traffic."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_sum_sum_96fb8a704ae4"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 32768
COLS = 128
VIEW_3D = (256, 128, 128)
FLAT_SHAPE = (ROWS, COLS)
REDUCE_SHAPE = (COLS,)
ATOMIC_BLOCK_M = 256
ATOMIC_BLOCK_N = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_outputs_kernel(out0_ptr, out1_ptr, out2_ptr, BLOCK_N: tl.constexpr):
        cols = tl.arange(0, BLOCK_N)
        tl.store(out0_ptr + cols, tl.zeros((BLOCK_N,), tl.float32))
        tl.store(out1_ptr + cols, tl.zeros((BLOCK_N,), tl.float32))
        tl.store(out2_ptr + cols, tl.zeros((BLOCK_N,), tl.float32))

    @triton.jit
    def _atomic_product_sum_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        scale_ptr,
        product_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        tile_m = tl.program_id(0)
        tile_n = tl.program_id(1)
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets = rows[:, None] * 128 + cols[None, :]

        a_vals = tl.load(a_ptr + offsets).to(tl.float32)
        b_vals = tl.load(b_ptr + offsets).to(tl.float32)
        c_vals = tl.load(c_ptr + offsets).to(tl.float32)
        add_vals = a_vals + b_vals

        scale_vals = tl.load(scale_ptr + cols).to(tl.float32)
        product_vals = add_vals * scale_vals[None, :]
        tl.store(product_ptr + offsets, product_vals)

        sum0 = tl.sum(add_vals, axis=0)
        sum1 = tl.sum(add_vals * c_vals, axis=0)
        sum2 = tl.sum(product_vals, axis=0)
        tl.atomic_add(out0_ptr + cols, sum0, sem="relaxed")
        tl.atomic_add(out1_ptr + cols, sum1, sem="relaxed")
        tl.atomic_add(out2_ptr + cols, sum2, sem="relaxed")


def _validate_shape_param(name: str, actual: object, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    mm_716: torch.Tensor,
    mm_718: torch.Tensor,
    arg589_1: torch.Tensor,
    arg8_1: torch.Tensor,
    shape0: object,
    shape1: object,
    shape2: object,
    shape3: object,
    shape4: object,
    shape5: object,
    shape6: object,
) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not (mm_716.is_cuda and mm_718.is_cuda and arg589_1.is_cuda and arg8_1.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if mm_716.dtype != torch.float32 or mm_718.dtype != torch.float32:
        raise TypeError("expected fp32 mm inputs")
    if arg589_1.dtype != torch.float32 or arg8_1.dtype != torch.float32:
        raise TypeError("expected fp32 multiplier inputs")
    for name, tensor in (
        ("mm_716", mm_716),
        ("mm_718", mm_718),
        ("arg589_1", arg589_1),
    ):
        if tuple(tensor.shape) != FLAT_SHAPE or tuple(tensor.stride()) != (COLS, 1):
            raise ValueError(f"{name} must be contiguous {FLAT_SHAPE}, got {tuple(tensor.shape)} stride {tuple(tensor.stride())}")
    if tuple(arg8_1.shape) != REDUCE_SHAPE or tuple(arg8_1.stride()) != (1,):
        raise ValueError(f"arg8_1 must be contiguous {REDUCE_SHAPE}, got {tuple(arg8_1.shape)} stride {tuple(arg8_1.stride())}")
    _validate_shape_param("_shape_param_0", shape0, VIEW_3D)
    _validate_shape_param("_shape_param_1", shape1, VIEW_3D)
    _validate_shape_param("_shape_param_2", shape2, REDUCE_SHAPE)
    _validate_shape_param("_shape_param_3", shape3, VIEW_3D)
    _validate_shape_param("_shape_param_4", shape4, REDUCE_SHAPE)
    _validate_shape_param("_shape_param_5", shape5, FLAT_SHAPE)
    _validate_shape_param("_shape_param_6", shape6, REDUCE_SHAPE)


def oracle_forward(inputs):
    """Compute the same four outputs as Repro.forward(*inputs)."""
    (
        mm_716,
        mm_718,
        arg589_1,
        arg8_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
    ) = inputs
    _validate_inputs(
        mm_716,
        mm_718,
        arg589_1,
        arg8_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
    )

    out0_base = torch.empty_strided((1, 1, COLS), (COLS, COLS, 1), device=mm_716.device, dtype=torch.float32)
    out1_base = torch.empty_strided((1, 1, COLS), (COLS, COLS, 1), device=mm_716.device, dtype=torch.float32)
    out3_base = torch.empty_strided((1, COLS), (COLS, 1), device=mm_716.device, dtype=torch.float32)
    product_base = torch.empty_strided(FLAT_SHAPE, (COLS, 1), device=mm_716.device, dtype=torch.float32)

    _zero_outputs_kernel[(1,)](
        out0_base,
        out1_base,
        out3_base,
        BLOCK_N=COLS,
        num_warps=4,
    )
    _atomic_product_sum_kernel[(triton.cdiv(ROWS, ATOMIC_BLOCK_M), triton.cdiv(COLS, ATOMIC_BLOCK_N))](
        mm_716,
        mm_718,
        arg589_1,
        arg8_1,
        product_base,
        out0_base,
        out1_base,
        out3_base,
        BLOCK_M=ATOMIC_BLOCK_M,
        BLOCK_N=ATOMIC_BLOCK_N,
        num_warps=8,
    )

    out0 = torch.ops.aten.view.default(out0_base, shape2)
    out1 = torch.ops.aten.view.default(out1_base, shape4)
    product = torch.ops.aten.view.default(product_base, shape5)
    out2 = torch.ops.aten.permute.default(product, [1, 0])
    out3 = torch.ops.aten.view.default(out3_base, shape6)
    return out0, out1, out2, out3


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
