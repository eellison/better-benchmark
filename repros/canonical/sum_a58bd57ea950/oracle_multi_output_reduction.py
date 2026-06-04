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


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1.0e-8)
    return float(torch.nan_to_num(diff, nan=0.0).max().item()), float(torch.nan_to_num(rel, nan=0.0).max().item())


def run_check(
    *,
    rtol: float,
    atol: float,
    block_m: int,
    block_c: int,
    num_warps: int,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = oracle_fused(*inputs, block_m=block_m, block_c=block_c, num_warps=num_warps)
        torch.cuda.synchronize()

    all_ok = True
    for index, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        max_abs, max_rel = _max_diff(got, ref)
        ok = shape_ok and dtype_ok and stride_ok and value_ok
        all_ok = all_ok and ok
        print(
            f"output[{index}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={value_ok} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )
    print(f"Correctness: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def _compile_with_config(
    module,
    inputs: tuple[object, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    *,
    warmup: int,
    rep: int,
    no_compile: bool,
    block_m: int,
    block_c: int,
    num_warps: int,
) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    mm_66, arg225_1, *shape_params = inputs
    _validate_inputs(mm_66, arg225_1, *shape_params)
    out_transposed, out_sum = _make_outputs(mm_66.device)

    timings: dict[str, float] = {}
    with torch.no_grad():
        _oracle_atomic_into(
            mm_66,
            arg225_1,
            out_transposed,
            out_sum,
            block_m=block_m,
            block_c=block_c,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: _oracle_atomic_into(
                mm_66,
                arg225_1,
                out_transposed,
                out_sum,
                block_m=block_m,
                block_c=block_c,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    timings["oracle_us"] = oracle_us
    logical_bytes = NUMEL * 4 * 3
    print(
        f"oracle_full_scope_silu_transpose_sum: {oracle_us:.3f} us "
        f"(block_m={block_m}, block_c={block_c}, num_warps={num_warps}, "
        f"logical_bw={logical_bytes / (oracle_us * 1.0e-6) / 1.0e12:.3f} TB/s)"
    )

    compile_times: list[tuple[str, float]] = []
    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            compiled = _compile_with_config(module, inputs, config)
            with torch.no_grad():
                compiled_us = triton.testing.do_bench(
                    lambda: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                ) * 1000.0
            timings[label] = compiled_us
            compile_times.append((label, compiled_us))
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    best_required_compile = min((us for _, us in compile_times), default=float("inf"))
    best_gate = min(best_required_compile, HISTORICAL_BEST_COMPILE_US)
    true_floor = oracle_us < best_gate
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    if best_required_compile != float("inf"):
        print(f"best required local compile_us: {best_required_compile:.3f} us")
    print(f"true_floor: {'yes' if true_floor else 'no (diagnosis-only)'}")
    print(
        json.dumps(
            {
                "repro_id": REPRO_ID,
                "oracle_us": round(oracle_us, 3),
                "best_required_compile_us": (
                    round(best_required_compile, 3)
                    if best_required_compile != float("inf")
                    else None
                ),
                "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
                "true_floor": true_floor,
                "classification": CLASSIFICATION,
            }
        )
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against eager repro.py")
    parser.add_argument("--bench", action="store_true", help="benchmark oracle and required compile configs")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-c", type=int, default=DEFAULT_BLOCK_C)
    parser.add_argument("--num-warps", type=int, default=DEFAULT_NUM_WARPS)
    parser.add_argument("--no-compile", action="store_true", help="benchmark only the Triton oracle")
    args = parser.parse_args()

    if args.rep <= 0:
        raise ValueError("--rep must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        block_m=args.block_m,
        block_c=args.block_c,
        num_warps=args.num_warps,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
            block_m=args.block_m,
            block_c=args.block_c,
            num_warps=args.num_warps,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
