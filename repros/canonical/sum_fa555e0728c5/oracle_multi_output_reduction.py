"""
Full-scope Triton oracle for sum_fa555e0728c5 (MobileBERT masked transpose + sum).

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle consumes the same
`mm_708`, boolean mask, scalar fill value, and shape parameters as repro.py,
materializes the returned `[512, 32768]` transpose-view backing storage while
accumulating the sibling `[512]` column sum from the same streamed `where`
producer, and returns outputs with the same dtypes, shapes, and strides.
Inductor cannot expose a meaningful remaining optimization for this captured
shape today because the best compiled variants already lower the full scope to
two bandwidth-dominated kernels with effectively the same memory traffic: read
`mm`, read the bool mask, write the full returned tensor, and finalize small
column partials. The fix class is BANDWIDTH_BOUND; no scheduler change is
justified unless a future full-scope oracle beats both required local compile
configs and the historical queue best.
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


REPRO_ID = "sum_fa555e0728c5"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 32768
C = 512
HISTORICAL_BEST_COMPILE_US = 33.76000002026558

DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 64
DEFAULT_FINAL_BLOCK_C = 16

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

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _where_store_partial_sum_kernel(
    mm_ptr,
    mask_ptr,
    full_ptr,
    out_base_ptr,
    partial_ptr,
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

    mm_values = tl.load(mm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mask_values = tl.load(mask_ptr + offsets, mask=active, other=0)
    full_value = tl.load(full_ptr).to(tl.float32)
    values = tl.where(mask_values, full_value, mm_values)

    tl.store(out_base_ptr + offsets, values, mask=active)
    partial = tl.sum(tl.where(active, values, 0.0), axis=0)
    tl.store(partial_ptr + row_block * C_ + cols, partial, mask=cols < C_)


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    active = (row_blocks[:, None] < NUM_ROW_BLOCKS) & (cols[None, :] < C_)
    offsets = row_blocks[:, None] * C_ + cols[None, :]

    partials = tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    tl.store(sum_ptr + cols, tl.sum(partials, axis=0), mask=cols < C_)


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
    if len(inputs) != 6:
        raise ValueError(f"expected 6 repro inputs, got {len(inputs)}")

    mm_708, arg1313_1, full_1, shape0, shape1, shape2 = inputs
    if not isinstance(mm_708, torch.Tensor) or not isinstance(arg1313_1, torch.Tensor):
        raise TypeError("first two inputs must be tensors")
    if not isinstance(full_1, torch.Tensor):
        raise TypeError("third input must be the scalar fill tensor")
    if tuple(shape0) != (256, 128, 512):
        raise ValueError(f"unexpected first shape parameter: {shape0}")
    if tuple(shape1) != (ROWS, C):
        raise ValueError(f"unexpected second shape parameter: {shape1}")
    if tuple(shape2) != (C,):
        raise ValueError(f"unexpected third shape parameter: {shape2}")

    if mm_708.shape != (ROWS, C):
        raise ValueError(f"unexpected mm_708 shape: {tuple(mm_708.shape)}")
    if arg1313_1.shape != (256, 128, C):
        raise ValueError(f"unexpected mask shape: {tuple(arg1313_1.shape)}")
    if full_1.shape != ():
        raise ValueError(f"unexpected full_1 shape: {tuple(full_1.shape)}")
    if mm_708.stride() != (C, 1):
        raise ValueError(f"unexpected mm_708 stride: {mm_708.stride()}")
    if arg1313_1.stride() != (128 * C, C, 1):
        raise ValueError(f"unexpected mask stride: {arg1313_1.stride()}")
    if mm_708.dtype != torch.float32 or full_1.dtype != torch.float32:
        raise ValueError("expected float32 mm_708 and full_1")
    if arg1313_1.dtype != torch.bool:
        raise ValueError(f"expected bool mask, got {arg1313_1.dtype}")
    if any(t.device.type != "cuda" for t in (mm_708, arg1313_1, full_1)):
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")


def oracle_fused(
    mm_708: torch.Tensor,
    arg1313_1: torch.Tensor,
    full_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    final_block_c: int = DEFAULT_FINAL_BLOCK_C,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = (mm_708, arg1313_1, full_1, _shape_param_0, _shape_param_1, _shape_param_2)
    _validate_inputs(inputs)

    num_row_blocks = triton.cdiv(ROWS, block_m)
    out_base = torch.empty((ROWS, C), device=mm_708.device, dtype=torch.float32)
    partial = torch.empty((num_row_blocks, C), device=mm_708.device, dtype=torch.float32)
    out_sum = torch.empty((C,), device=mm_708.device, dtype=torch.float32)

    _where_store_partial_sum_kernel[(num_row_blocks, triton.cdiv(C, block_n))](
        mm_708,
        arg1313_1,
        full_1,
        out_base,
        partial,
        ROWS_=ROWS,
        C_=C,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    _finalize_sum_kernel[(triton.cdiv(C, final_block_c),)](
        partial,
        out_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=final_block_c,
        num_warps=8,
    )

    return out_base.permute(1, 0), out_sum


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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
