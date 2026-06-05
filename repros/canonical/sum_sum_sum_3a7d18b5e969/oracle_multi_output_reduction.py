"""
Full-scope Triton oracle for sum_sum_sum_3a7d18b5e969 (MobileBERT
multi-output reduction tail).

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle consumes
the same four `[32768, 512]`/`[256, 128, 512]` additive producers, the
elementwise `arg611_1` multiplier, the per-column `arg27_1` multiplier, and
the same shape parameters as `repro.py`; it materializes the returned
`[512, 32768]` transpose-view backing storage while accumulating
`sum(add_tensor_2)` and `sum(add_tensor_2 * arg611_1)` from the same streamed
producer, then derives `sum(add_tensor_2 * arg27_1)` as
`sum(add_tensor_2) * arg27_1` in the finalizer. Inductor cannot do this today
because the FX graph exposes the shared producer, two large multiplies, three
column reductions, and the required transpose side output as independent
view/pointwise/sum/permute nodes, and the scheduler does not prove the
per-column multiply can be moved after the reduction while fusing the remaining
same-axis reductions with the side-output store; the fix is an
ALGEBRAIC_ELIMINATION pass for linear per-column reductions combined with a
dependent multi-output reduction template that carries the materialized
transpose side output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_sum_sum_3a7d18b5e969"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 256
SEQ = 128
ROWS = BATCH * SEQ
C = 512
HISTORICAL_BEST_COMPILE_US = 85.88799834251404

DEFAULT_BLOCK_M = 256
DEFAULT_BLOCK_N = 64
DEFAULT_FINAL_BLOCK_C = 32
DEFAULT_STORE_WARPS = 8
DEFAULT_FINAL_WARPS = 8

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _store_and_partial_reduce_kernel(
    mm_684_ptr,
    mul_494_ptr,
    mm_690_ptr,
    mm_692_ptr,
    arg611_ptr,
    arg27_ptr,
    out_base_ptr,
    partial_sum_add_ptr,
    partial_sum_weighted_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    active = (rows[:, None] < ROWS_) & (cols[None, :] < C_)
    offsets = rows[:, None] * C_ + cols[None, :]

    add_value = (
        tl.load(mul_494_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm_684_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm_690_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm_692_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    )
    arg611 = tl.load(arg611_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    arg27 = tl.load(arg27_ptr + cols, mask=cols < C_, other=0.0).to(tl.float32)
    out_value = add_value * arg27[None, :]

    tl.store(out_base_ptr + offsets, out_value, mask=active)

    partial_sum_add = tl.sum(tl.where(active, add_value, 0.0), axis=0)
    partial_sum_weighted = tl.sum(tl.where(active, add_value * arg611, 0.0), axis=0)
    partial_offsets = row_block * C_ + cols
    tl.store(partial_sum_add_ptr + partial_offsets, partial_sum_add, mask=cols < C_)
    tl.store(
        partial_sum_weighted_ptr + partial_offsets,
        partial_sum_weighted,
        mask=cols < C_,
    )


@triton.jit
def _finalize_partials_kernel(
    partial_sum_add_ptr,
    partial_sum_weighted_ptr,
    arg27_ptr,
    out_sum_add_ptr,
    out_sum_weighted_ptr,
    out_sum_scaled_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    active = (row_blocks[:, None] < NUM_ROW_BLOCKS) & (cols[None, :] < C_)
    offsets = row_blocks[:, None] * C_ + cols[None, :]

    partial_sum_add = tl.load(
        partial_sum_add_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    partial_sum_weighted = tl.load(
        partial_sum_weighted_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    total_add = tl.sum(partial_sum_add, axis=0)
    total_weighted = tl.sum(partial_sum_weighted, axis=0)
    arg27 = tl.load(arg27_ptr + cols, mask=cols < C_, other=0.0).to(tl.float32)
    col_mask = cols < C_

    tl.store(out_sum_add_ptr + cols, total_add, mask=col_mask)
    tl.store(out_sum_weighted_ptr + cols, total_weighted, mask=col_mask)
    tl.store(out_sum_scaled_ptr + cols, total_add * arg27, mask=col_mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        x.cuda() if isinstance(x, torch.Tensor) and x.device.type != "cuda" else x
        for x in module.make_inputs()
    )


def _validate_inputs(inputs: tuple[object, ...]) -> None:
    if len(inputs) != 13:
        raise ValueError(f"expected 13 repro inputs, got {len(inputs)}")

    (
        mm_684,
        mul_494,
        mm_690,
        mm_692,
        arg611_1,
        arg27_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
    ) = inputs
    tensor_inputs = (mm_684, mul_494, mm_690, mm_692, arg611_1, arg27_1)
    if any(not isinstance(t, torch.Tensor) for t in tensor_inputs):
        raise TypeError("first six inputs must be tensors")
    if any(t.device.type != "cuda" for t in tensor_inputs):
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if any(t.dtype != torch.float32 for t in tensor_inputs):
        raise ValueError("expected all tensor inputs to be float32")

    expected_3d = (BATCH, SEQ, C)
    expected_2d = (ROWS, C)
    if mm_684.shape != expected_2d or mm_690.shape != expected_2d:
        raise ValueError("unexpected mm input shape")
    if mm_692.shape != expected_2d:
        raise ValueError(f"unexpected mm_692 shape: {tuple(mm_692.shape)}")
    if mul_494.shape != expected_3d or arg611_1.shape != expected_3d:
        raise ValueError("unexpected 3D input shape")
    if arg27_1.shape != (C,):
        raise ValueError(f"unexpected arg27_1 shape: {tuple(arg27_1.shape)}")

    if mm_684.stride() != (C, 1) or mm_690.stride() != (C, 1):
        raise ValueError("unexpected mm input stride")
    if mm_692.stride() != (C, 1):
        raise ValueError(f"unexpected mm_692 stride: {mm_692.stride()}")
    if mul_494.stride() != (SEQ * C, C, 1):
        raise ValueError(f"unexpected mul_494 stride: {mul_494.stride()}")
    if arg611_1.stride() != (SEQ * C, C, 1):
        raise ValueError(f"unexpected arg611_1 stride: {arg611_1.stride()}")
    if arg27_1.stride() != (1,):
        raise ValueError(f"unexpected arg27_1 stride: {arg27_1.stride()}")

    expected_shapes = (
        expected_3d,
        expected_3d,
        expected_3d,
        (C,),
        (C,),
        expected_2d,
        (C,),
    )
    for idx, (shape_param, expected) in enumerate(
        zip((shape0, shape1, shape2, shape3, shape4, shape5, shape6), expected_shapes)
    ):
        if tuple(shape_param) != expected:
            raise ValueError(f"unexpected shape parameter {idx}: {shape_param}")


def oracle_fused(
    mm_684: torch.Tensor,
    mul_494: torch.Tensor,
    mm_690: torch.Tensor,
    mm_692: torch.Tensor,
    arg611_1: torch.Tensor,
    arg27_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    final_block_c: int = DEFAULT_FINAL_BLOCK_C,
    store_warps: int = DEFAULT_STORE_WARPS,
    final_warps: int = DEFAULT_FINAL_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = (
        mm_684,
        mul_494,
        mm_690,
        mm_692,
        arg611_1,
        arg27_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    )
    _validate_inputs(inputs)

    num_row_blocks = triton.cdiv(ROWS, block_m)
    out_base = torch.empty((ROWS, C), device=mm_684.device, dtype=torch.float32)
    partial_sum_add = torch.empty(
        (num_row_blocks, C), device=mm_684.device, dtype=torch.float32
    )
    partial_sum_weighted = torch.empty(
        (num_row_blocks, C), device=mm_684.device, dtype=torch.float32
    )
    out_sum_add = torch.empty((C,), device=mm_684.device, dtype=torch.float32)
    out_sum_weighted = torch.empty((C,), device=mm_684.device, dtype=torch.float32)
    out_sum_scaled = torch.empty((C,), device=mm_684.device, dtype=torch.float32)

    _store_and_partial_reduce_kernel[
        (num_row_blocks, triton.cdiv(C, block_n))
    ](
        mm_684,
        mul_494,
        mm_690,
        mm_692,
        arg611_1,
        arg27_1,
        out_base,
        partial_sum_add,
        partial_sum_weighted,
        ROWS_=ROWS,
        C_=C,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=store_warps,
    )
    _finalize_partials_kernel[(triton.cdiv(C, final_block_c),)](
        partial_sum_add,
        partial_sum_weighted,
        arg27_1,
        out_sum_add,
        out_sum_weighted,
        out_sum_scaled,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=final_block_c,
        num_warps=final_warps,
    )

    return out_sum_add, out_sum_weighted, out_base.permute(1, 0), out_sum_scaled


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
