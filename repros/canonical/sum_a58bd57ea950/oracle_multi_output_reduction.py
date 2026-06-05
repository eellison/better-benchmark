"""
Full-scope Triton oracle for sum_a58bd57ea950.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the
complete MobileViT SiLU-gradient return tuple from repro.py, including the
pointwise `mm * sigmoid(arg) * (arg * (1 - sigmoid(arg)) + 1)` producer, the
returned f32[288, 131072] transpose view with stride (1, 288), and the returned
f32[288] column sum, by streaming the producer once through a Triton tile that
stores the side output and accumulates the reduction. Inductor cannot express
this exact hand-written schedule today because its generic scheduler does not
model a materialized transpose side-output and a same-producer column reduction
as one explicitly tiled multi-output reduction with pre-zeroed atomic
accumulators. The Inductor change that would fix the scheduler-fusion
hypothesis is to teach the multi-output reduction template to emit a fused
producer that writes required dense side outputs while reducing the same values;
if this oracle does not beat both required compile configs and the historical
best gate, the repro is diagnosis-only/BANDWIDTH_BOUND rather than a true floor.
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


REPRO_ID = "sum_a58bd57ea950"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 131_072
C = 288
NUMEL = ROWS * C
HISTORICAL_BEST_COMPILE_US = 87.80799806118011

DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_C = 32
DEFAULT_NUM_WARPS = 4

CLASSIFICATION = "BANDWIDTH_BOUND"

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
def _zero_sum_kernel(
    out_sum_ptr,
    C_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_C)
    tl.store(out_sum_ptr + offsets, tl.zeros([BLOCK_C], dtype=tl.float32), mask=offsets < C_)


@triton.jit
def _silu_transpose_sum_atomic_kernel(
    mm_ptr,
    arg_ptr,
    out_transposed_ptr,
    out_sum_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (row_offsets[:, None] < ROWS_) & (col_offsets[None, :] < C_)
    offsets = row_offsets[:, None] * C_ + col_offsets[None, :]

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    arg = tl.load(arg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (tl.exp(-arg) + 1.0)
    value = mm * sigmoid * (arg * (1.0 - sigmoid) + 1.0)

    # out_transposed has logical shape [C, ROWS] and stride [1, C], so its
    # physical storage offset is the original row-major [ROWS, C] offset.
    tl.store(out_transposed_ptr + offsets, value, mask=mask)

    sums = tl.sum(tl.where(mask, value, 0.0), axis=0)
    tl.atomic_add(out_sum_ptr + col_offsets, sums, sem="relaxed", mask=col_offsets < C_)


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
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def get_inputs() -> tuple[object, ...]:
    return make_inputs()


def get_repro_instance() -> torch.nn.Module:
    return _load_repro_module().Repro().cuda()


def _validate_tiling(block_m: int, block_c: int, num_warps: int) -> None:
    for name, value in (("block_m", block_m), ("block_c", block_c)):
        if value <= 0 or (value & (value - 1)) != 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive power of two")
    if num_warps <= 0:
        raise ValueError("--num-warps must be positive")


def _validate_inputs(
    mm_66: torch.Tensor,
    arg225_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> None:
    expected_3d = [512, 256, C]
    expected_2d = [ROWS, C]
    expected_1d = [C]
    if _shape_param_0 != expected_3d or _shape_param_1 != expected_3d:
        raise ValueError("unexpected 3D view shape parameters")
    if _shape_param_2 != expected_2d or _shape_param_3 != expected_1d:
        raise ValueError("unexpected output view shape parameters")
    for name, tensor in (("mm_66", mm_66), ("arg225_1", arg225_1)):
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != (ROWS, C):
            raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} has unexpected dtype {tensor.dtype}")
        if tensor.stride() != (C, 1):
            raise ValueError(f"{name} must be contiguous row-major, got stride {tensor.stride()}")


def _make_outputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    out_transposed = torch.empty_strided((C, ROWS), (1, C), device=device, dtype=torch.float32)
    out_sum = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    return out_transposed, out_sum


def _oracle_atomic_into(
    mm_66: torch.Tensor,
    arg225_1: torch.Tensor,
    out_transposed: torch.Tensor,
    out_sum: torch.Tensor,
    *,
    block_m: int,
    block_c: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_tiling(block_m, block_c, num_warps)
    if out_transposed.shape != (C, ROWS) or out_transposed.stride() != (1, C):
        raise ValueError("unexpected transposed output layout")
    if out_sum.shape != (C,) or out_sum.stride() != (1,):
        raise ValueError("unexpected sum output layout")

    _zero_sum_kernel[(1,)](
        out_sum,
        C_=C,
        BLOCK_C=triton.next_power_of_2(C),
        num_warps=1,
    )
    grid = (triton.cdiv(ROWS, block_m), triton.cdiv(C, block_c))
    _silu_transpose_sum_atomic_kernel[grid](
        mm_66,
        arg225_1,
        out_transposed,
        out_sum,
        ROWS_=ROWS,
        C_=C,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    return out_transposed, out_sum


def oracle_fused(
    mm_66: torch.Tensor,
    arg225_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_c: int = DEFAULT_BLOCK_C,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(mm_66, arg225_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3)
    out_transposed, out_sum = _make_outputs(mm_66.device)
    return _oracle_atomic_into(
        mm_66,
        arg225_1,
        out_transposed,
        out_sum,
        block_m=block_m,
        block_c=block_c,
        num_warps=num_warps,
    )


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, torch.Tensor]:
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    model = get_repro_instance()
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
