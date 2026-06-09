"""
Full-scope Triton oracle for sum_sum_25d0782111b0.

Gap diagnosis: Classification: BANDWIDTH_BOUND. This oracle consumes the same
eight inputs and returns the same tensor/vector pair as repro.py, but it keeps
each channel inside one Triton program: it recomputes the cheap where/centered
pointwise expression, accumulates the two sibling reductions locally, then writes
the dependent BN-backward tensor epilogue and scale-gradient vector without
global reduction scratch buffers. Inductor already emits a near-optimal
multi-output reduction for this small 8192-element/channel case; it cannot use
this exact single-program-per-channel schedule generically because larger
channels need split reductions for occupancy and because the scheduler lacks a
profit model that trades parallel reduction depth against launch/scratch traffic
for the dependent full-tensor epilogue. The gap class is BANDWIDTH_BOUND: this
artifact is diagnosis-only unless it beats both required compile configs and the
historical best compile timing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_ID = "sum_sum_25d0782111b0"

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
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 384
H = 8
W = 8
HW = H * W
TOTAL_K = N * HW
NUMEL = N * C * HW
REDUCE_SCALE = 0.0001220703125
HISTORICAL_BEST_COMPILE_US = 16.48000068962574

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


def _load_repro_module():
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs():
    mod = _load_repro_module()
    if hasattr(mod, "make_inputs"):
        return mod.make_inputs()
    if hasattr(mod, "_default_make_inputs"):
        return mod._default_make_inputs()
    raise RuntimeError("Repro has no make_inputs or _default_make_inputs")


def get_repro_instance():
    mod = _load_repro_module()
    return mod.Repro()


if triton is not None:

    @triton.jit
    def _full_scope_channel_kernel(
        getitem_42_ptr,
        getitem_45_ptr,
        arg491_ptr,
        full_1_ptr,
        arg489_ptr,
        arg549_ptr,
        arg490_ptr,
        arg191_ptr,
        out_tensor_ptr,
        out_vector_ptr,
        BLOCK_K: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
    ):
        c = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        full_value = tl.load(full_1_ptr).to(tl.float32)
        mean = tl.load(arg549_ptr + c).to(tl.float32)

        sum_where = tl.full((), 0.0, tl.float32)
        sum_where_centered = tl.full((), 0.0, tl.float32)

        for tile in tl.static_range(0, NUM_K_BLOCKS):
            k = tile * BLOCK_K + offs
            n = k // HW_
            hw = k - n * HW_
            elem_offsets = n * (C_ * HW_) + c * HW_ + hw

            lhs = tl.load(getitem_42_ptr + elem_offsets).to(tl.float32)
            rhs = tl.load(getitem_45_ptr + elem_offsets).to(tl.float32)
            mask_source = tl.load(arg491_ptr + elem_offsets).to(tl.float32)
            where_value = tl.where(mask_source <= 0.0, full_value, lhs + rhs)
            centered = tl.load(arg489_ptr + elem_offsets).to(tl.float32) - mean

            sum_where += tl.sum(where_value, axis=0)
            sum_where_centered += tl.sum(where_value * centered, axis=0)

        invstd = tl.load(arg490_ptr + c).to(tl.float32)
        gamma_grad_scale = tl.load(arg191_ptr + c).to(tl.float32)
        variance_term = sum_where_centered * REDUCE_SCALE_ * invstd * invstd
        mean_term = sum_where * REDUCE_SCALE_
        output_scale = invstd * gamma_grad_scale
        tl.store(out_vector_ptr + c, sum_where_centered * invstd)

        for tile in tl.static_range(0, NUM_K_BLOCKS):
            k = tile * BLOCK_K + offs
            n = k // HW_
            hw = k - n * HW_
            elem_offsets = n * (C_ * HW_) + c * HW_ + hw

            lhs = tl.load(getitem_42_ptr + elem_offsets).to(tl.float32)
            rhs = tl.load(getitem_45_ptr + elem_offsets).to(tl.float32)
            mask_source = tl.load(arg491_ptr + elem_offsets).to(tl.float32)
            where_value = tl.where(mask_source <= 0.0, full_value, lhs + rhs)
            centered = tl.load(arg489_ptr + elem_offsets).to(tl.float32) - mean
            out = (where_value - centered * variance_term - mean_term) * output_scale
            tl.store(out_tensor_ptr + elem_offsets, out)


@oracle_impl(hardware="H100", shapes="(T([128, 384, 8, 8], f32), T([128, 384, 8, 8], f32), T([128, 384, 8, 8], f32), T([], f32), T([128, 384, 8, 8], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))")
def oracle_forward(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        getitem_42,
        getitem_45,
        arg491_1,
        full_1,
        arg489_1,
        arg549_1,
        arg490_1,
        arg191_1,
    ) = inputs

    assert getitem_42.shape == (N, C, H, W)
    assert getitem_45.shape == (N, C, H, W)
    assert arg491_1.shape == (N, C, H, W)
    assert arg489_1.shape == (N, C, H, W)
    assert arg549_1.shape == (1, C, 1, 1)
    assert arg490_1.shape == (C,)
    assert arg191_1.shape == (C,)
    assert getitem_42.is_contiguous()
    assert getitem_45.is_contiguous()
    assert arg491_1.is_contiguous()
    assert arg489_1.is_contiguous()
    assert arg549_1.is_contiguous()
    assert arg490_1.is_contiguous()
    assert arg191_1.is_contiguous()

    out_tensor = torch.empty_like(getitem_42)
    out_vector = torch.empty_like(arg490_1)
    block_k = 8192
    _full_scope_channel_kernel[(C,)](
        getitem_42,
        getitem_45,
        arg491_1,
        full_1,
        arg489_1,
        arg549_1,
        arg490_1,
        arg191_1,
        out_tensor,
        out_vector,
        BLOCK_K=block_k,
        NUM_K_BLOCKS=TOTAL_K // block_k,
        C_=C,
        HW_=HW,
        REDUCE_SCALE_=REDUCE_SCALE,
        num_warps=16,
    )
    return out_tensor, out_vector


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
