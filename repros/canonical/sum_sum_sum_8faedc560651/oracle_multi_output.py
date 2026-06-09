"""
Oracle for sum_sum_sum_8faedc560651.

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full
Repro.forward return tuple in one Triton program family, preserving the
`where(mask, 0, mm)` semantics, the per-`[64, 32]` grouped 16-wide summaries,
the full `[64, 512, 1, 1]` pointwise output, and both returned `[512]` batch
reductions, whereas Inductor lowers the decomposed singleton reductions,
grouped reductions, dependent pointwise epilogue, and sibling batch reductions
as multiple generic kernels over materialized intermediates; Inductor cannot do
this today because the scheduler does not form a single grouped multi-output
reduction node whose row-local summaries feed both a full tensor side output and
two compatible column reductions; the fix is SCHEDULER_FUSION: add dependent
multi-output reduction scheduling for fixed small grouped reductions so the
shared masked producer is read once and all three outputs are finalized in the
same generated kernel.
"""
from __future__ import annotations

import argparse
import json
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

from oracle_harness import (
    oracle_impl,
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

BATCH = 64
GROUPS = 32
GROUP_WIDTH = 16
FEATURES = GROUPS * GROUP_WIDTH

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


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _full_scope_kernel(
        mm_ptr,
        mask_ptr,
        arg117_ptr,
        arg41_ptr,
        arg119_ptr,
        arg118_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        FEATURES_: tl.constexpr,
        GROUPS_: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        group = tl.program_id(0)
        b = tl.arange(0, BLOCK_B)
        k = tl.arange(0, BLOCK_K)
        feature = group * BLOCK_K + k
        offsets = b[:, None] * FEATURES_ + feature[None, :]
        group_offsets = b * GROUPS_ + group

        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        mask_value = tl.load(mask_ptr + offsets)
        arg117 = tl.load(arg117_ptr + offsets).to(tl.float32)
        arg41 = tl.load(arg41_ptr + feature).to(tl.float32)
        arg119 = tl.load(arg119_ptr + group_offsets).to(tl.float32)
        arg118 = tl.load(arg118_ptr + group_offsets).to(tl.float32)

        masked_mm = tl.where(mask_value, 0.0, mm)
        weighted_masked = masked_mm * arg41[None, :]
        weighted_arg117 = weighted_masked * arg117

        sum_arg117_weighted = tl.sum(weighted_arg117, axis=1)
        sum_weighted = tl.sum(weighted_masked, axis=1)

        delta = sum_weighted * arg118 - sum_arg117_weighted
        arg119_sq = arg119 * arg119
        grouped_scale = delta * arg119_sq * arg119 * 0.0625
        grouped_bias = -(grouped_scale * arg118) - (sum_weighted * arg119 * 0.0625)

        out0 = (
            masked_mm * arg119[:, None] * arg41[None, :]
            + arg117 * grouped_scale[:, None]
            + grouped_bias[:, None]
        )
        out1 = tl.sum(masked_mm * (arg117 - arg118[:, None]) * arg119[:, None], axis=0)
        out2 = tl.sum(masked_mm, axis=0)

        tl.store(out0_ptr + offsets, out0)
        tl.store(out1_ptr + feature, out1)
        tl.store(out2_ptr + feature, out2)


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _validate_inputs(inputs: tuple[Any, ...]) -> None:
    _require_triton()
    if len(inputs) != 19:
        raise ValueError(f"expected 19 Repro inputs, got {len(inputs)}")

    (
        mm,
        arg121_1,
        arg117_1,
        arg41_1,
        arg119_1,
        arg118_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
        shape7,
        shape8,
        shape9,
        shape10,
        shape11,
        shape12,
    ) = inputs

    expected_tensors = {
        "mm": (mm, (BATCH, FEATURES), torch.float32),
        "arg121_1": (arg121_1, (BATCH, FEATURES, 1, 1), torch.bool),
        "arg117_1": (arg117_1, (BATCH, FEATURES, 1, 1), torch.float32),
        "arg41_1": (arg41_1, (FEATURES,), torch.float32),
        "arg119_1": (arg119_1, (BATCH, GROUPS), torch.float32),
        "arg118_1": (arg118_1, (BATCH, GROUPS), torch.float32),
    }
    for name, (tensor, shape, dtype) in expected_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    expected_shapes = [
        [BATCH, FEATURES, 1, 1],
        [BATCH, FEATURES, 1, 1],
        [BATCH, FEATURES, 1],
        [BATCH, FEATURES, 1],
        [BATCH, GROUPS, GROUP_WIDTH],
        [BATCH, GROUPS, GROUP_WIDTH],
        [1, GROUPS, GROUP_WIDTH],
        [BATCH, GROUPS, GROUP_WIDTH, 1],
        [BATCH, GROUPS, GROUP_WIDTH, 1],
        [BATCH, FEATURES, 1, 1],
        [BATCH, GROUPS, GROUP_WIDTH],
        [BATCH, GROUPS, GROUP_WIDTH],
        [FEATURES],
    ]
    actual_shapes = [
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
        shape7,
        shape8,
        shape9,
        shape10,
        shape11,
        shape12,
    ]
    for idx, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
        if list(actual) != expected:
            raise ValueError(f"unexpected _shape_param_{idx}: {actual!r}")


@oracle_impl(hardware="H100", shapes="(T([64, 512], f32), T([64, 512, 1, 1], b8), T([64, 512, 1, 1], f32), T([512], f32), T([64, 32], f32), T([64, 32], f32), S([64, 512, 1, 1]), S([64, 512, 1, 1]), S([64, 512, 1]), S([64, 512, 1]), S([64, 32, 16]), S([64, 32, 16]), S([1, 32, 16]), S([64, 32, 16, 1]), S([64, 32, 16, 1]), S([64, 512, 1, 1]), S([64, 32, 16]), S([64, 32, 16]), S([512]))")
def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward computation for the canonical shape."""
    inputs = tuple(inputs)
    _validate_inputs(inputs)
    mm, arg121_1, arg117_1, arg41_1, arg119_1, arg118_1, *_shape_params = inputs

    out0 = torch.empty_strided(
        (BATCH, FEATURES, 1, 1),
        (FEATURES, 1, 1, 1),
        device=mm.device,
        dtype=torch.float32,
    )
    out1 = torch.empty((FEATURES,), device=mm.device, dtype=torch.float32)
    out2 = torch.empty((FEATURES,), device=mm.device, dtype=torch.float32)

    _full_scope_kernel[(GROUPS,)](
        mm,
        arg121_1,
        arg117_1,
        arg41_1,
        arg119_1,
        arg118_1,
        out0,
        out1,
        out2,
        FEATURES_=FEATURES,
        GROUPS_=GROUPS,
        BLOCK_B=BATCH,
        BLOCK_K=GROUP_WIDTH,
        num_warps=8,
    )
    return out0, out1, out2


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [item for item in out if isinstance(item, torch.Tensor)]
    return []


def _strict_scope_check(instance: torch.nn.Module, inputs: tuple[Any, ...]) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    actual_list = _normalize_outputs(actual)
    if len(eager_list) != len(actual_list):
        print(
            f"  strict scope: FAIL output_count oracle={len(actual_list)} "
            f"eager={len(eager_list)}"
        )
        return False

    ok = True
    for idx, (got, expected) in enumerate(zip(actual_list, eager_list)):
        shape_ok = got.shape == expected.shape
        dtype_ok = got.dtype == expected.dtype
        stride_ok = got.stride() == expected.stride()
        ok = ok and shape_ok and dtype_ok and stride_ok
        print(
            f"  strict scope output {idx}: "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )
    print(f"  strict scope: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(instance: torch.nn.Module, inputs: tuple[Any, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(instance)
        with torch.no_grad():
            for _ in range(5):
                compiled(*inputs)
            torch.cuda.synchronize()
    return compiled


def _do_cuda_bench(fn, warmup: int, rep: int) -> float:
    _require_triton()
    with torch.no_grad():
        return triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0


def bench_required_configs(inputs: tuple[Any, ...], warmup: int, rep: int) -> dict[str, object]:
    """Benchmark the full-scope oracle and the required torch.compile configs."""
    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = _do_cuda_bench(lambda: oracle_forward(inputs), warmup, rep)

    print(f"oracle_fused full-scope grouped multi-output reduction: {oracle_us:.3f} us")

    compile_timings: dict[str, float] = {}
    for label, config in COMPILE_CONFIGS:
        instance = get_repro_instance()
        compiled = _compile_with_config(instance, inputs, config)
        with torch.no_grad():
            compile_us = _do_cuda_bench(lambda: compiled(*inputs), warmup, rep)
        compile_timings[label] = compile_us
        print(f"torch.compile {label}: {compile_us:.3f} us")

    best_compile_us = min(compile_timings.values())
    true_floor = oracle_us < best_compile_us
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_timings["coordinate_descent_tuning=True"], 3),
        "combo_compile_us": round(
            compile_timings[
                "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
                "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
                "triton.multi_kernel=3"
            ],
            3,
        ),
        "best_required_compile_us": round(best_compile_us, 3),
        "ratio": round(best_compile_us / oracle_us, 3) if oracle_us > 0.0 else 0.0,
        "status": "GOOD" if true_floor else "BAD_ORACLE",
        "true_floor": true_floor,
        "classification": "SCHEDULER_FUSION",
    }
    print(json.dumps(result))
    return result


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
        ok = _strict_scope_check(instance, inputs) and ok
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
                        "WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            bench_required_configs(tuple(inputs), warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
