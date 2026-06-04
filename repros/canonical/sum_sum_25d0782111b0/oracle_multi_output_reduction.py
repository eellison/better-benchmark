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
import math
import sys
import time
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_ID = "sum_sum_25d0782111b0"
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


def run_check(inputs, *, rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    instance = get_repro_instance()

    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        device = _get_device(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)

    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (e, o) in enumerate(zip(eager_list, oracle_list)):
        if e.shape != o.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(o.shape)} "
                f"eager={list(e.shape)}"
            )
            all_pass = False
            continue
        if e.stride() != o.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={list(o.stride())} "
                f"eager={list(e.stride())}"
            )
            all_pass = False
            continue
        if e.dtype != o.dtype:
            print(f"  output {i}: WARNING dtype mismatch oracle={o.dtype} eager={e.dtype}")

        e_f32 = e.float()
        o_f32 = o.float()
        max_diff = (e_f32 - o_f32).abs().max().item()
        ok = torch.allclose(e_f32, o_f32, atol=atol, rtol=rtol)

        status = "PASS" if ok else "FAIL"
        print(
            f"  output {i}: {status} (shape={list(e.shape)} dtype={e.dtype} "
            f"stride={list(e.stride())} max_diff={max_diff:.2e})"
        )
        if not ok:
            all_pass = False

    return all_pass


def run_bench(inputs, *, warmup: int = 10, rep: int = 50, no_compile: bool = False) -> dict:
    device = _get_device(inputs)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Triton oracle benchmarking")

    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = _do_bench(lambda: oracle_forward(inputs), device, warmup=warmup, rep=rep)

    compile_results: dict[str, float] = {}
    if not no_compile:
        for label, config in COMPILE_CONFIGS:
            instance = get_repro_instance()
            compiled = _compile_with_config(instance, inputs, config)
            with torch.no_grad():
                compile_results[label] = _do_bench(
                    lambda: compiled(*inputs),
                    device,
                    warmup=warmup,
                    rep=rep,
                )

    measured_best = min(compile_results.values()) if compile_results else math.inf
    comparison_us = min(measured_best, HISTORICAL_BEST_COMPILE_US)
    ratio = comparison_us / oracle_us if oracle_us > 0 else 0.0
    valid_floor = oracle_us < comparison_us
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(comparison_us, 3),
        "ratio": round(ratio, 3),
        "status": "GOOD" if valid_floor else "BAD_ORACLE",
        "valid_floor": valid_floor,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "compile_configs_us": {k: round(v, 3) for k, v in compile_results.items()},
    }
    print(json.dumps(result))
    return result


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config), torch.no_grad():
        compiled = torch.compile(model)
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _normalize_outputs(out) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                result.extend(_normalize_outputs(item))
        return result
    return []


def _get_device(inputs) -> torch.device:
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            return inp.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _do_bench(fn, device: torch.device, warmup: int = 10, rep: int = 50) -> float:
    if triton is not None and device.type == "cuda":
        from triton.testing import do_bench

        return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    best_us = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best_us = min(best_us, elapsed * 1_000_000.0)
    return best_us


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=50, help="Repetitions for benchmark")
    parser.add_argument("--no-compile", action="store_true", help="Only benchmark the Triton oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = run_check(inputs, rtol=args.rtol, atol=args.atol)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        result = run_bench(inputs, warmup=args.warmup, rep=args.rep, no_compile=args.no_compile)
        if result["status"] == "BAD_ORACLE":
            print(
                "WARNING: oracle is not faster than the required compile configs "
                "and historical best compile timing; keep this artifact diagnosis-only."
            )


if __name__ == "__main__":
    main()
