"""
Oracle for mean_453dc6e89dd9

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full captured `cat -> BN-inference affine -> ReLU -> spatial mean -> view` scope in one Triton kernel, reading directly from the two concat inputs and storing only the final f32 `[128,184]` tensor, whereas Inductor currently treats the channel concat as a materialized producer boundary before the downstream normalization/reduction schedule; Inductor cannot do this today because its scheduler does not represent `aten.cat` as a virtual multi-source layout that can feed a fused reduction consumer with per-channel source selection; the fix is SCHEDULER_FUSION: allow fixed-shape concat producers to be inlined into downstream pointwise/reduction kernels instead of forcing a dense concatenated intermediate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
C0 = 16
C1 = 168
CHANNELS = C0 + C1
SPATIAL = 16
EPS = 1.0e-5
BLOCK_N = 4
BLOCK_C = 16
CLASSIFICATION = "SCHEDULER_FUSION"
TRUE_FLOOR = True


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _cat_bn_relu_spatial_mean_kernel(
        conv_ptr,
        cat_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        BLOCK_BATCH: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        eps: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        c_block = tl.program_id(1)
        row_offsets = tl.arange(0, BLOCK_ROWS)
        n_offsets = n_block * BLOCK_BATCH + row_offsets // BLOCK_CHANNELS
        c_offsets = c_block * BLOCK_CHANNELS + row_offsets % BLOCK_CHANNELS
        hw_offsets = tl.arange(0, BLOCK_HW)

        row_mask = (n_offsets < 128) & (c_offsets < 184)
        use_conv = c_offsets < 16

        conv_offsets = (n_offsets[:, None] * 16 + c_offsets[:, None]) * 16 + hw_offsets[None, :]
        cat_offsets = (n_offsets[:, None] * 168 + (c_offsets[:, None] - 16)) * 16 + hw_offsets[None, :]
        valid = row_mask[:, None]

        conv_values = tl.load(conv_ptr + conv_offsets, mask=valid & use_conv[:, None], other=0.0)
        cat_values = tl.load(cat_ptr + cat_offsets, mask=valid & (~use_conv)[:, None], other=0.0)
        x = tl.where(use_conv[:, None], conv_values, cat_values).to(tl.float32)

        mean = tl.load(mean_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=row_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * tl.rsqrt(var[:, None] + eps) * weight[:, None] + bias[:, None]
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        reduced = tl.sum(y, axis=1) * 0.0625

        out_offsets = n_offsets * 184 + c_offsets
        tl.store(out_ptr + out_offsets, reduced, mask=row_mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    convolution_51, cat_22, mean, var, weight, bias, view_shape = inputs
    conv = _require_tensor("convolution_51", convolution_51, (BATCH, C0, 4, 4), (C0 * SPATIAL, SPATIAL, 4, 1))
    cat = _require_tensor("cat_22", cat_22, (BATCH, C1, 4, 4), (C1 * SPATIAL, SPATIAL, 4, 1))
    mean_t = _require_tensor("arg258_1", mean, (CHANNELS,), (1,))
    var_t = _require_tensor("arg259_1", var, (CHANNELS,), (1,))
    weight_t = _require_tensor("arg260_1", weight, (CHANNELS,), (1,))
    bias_t = _require_tensor("arg261_1", bias, (CHANNELS,), (1,))

    if list(view_shape) != [BATCH, CHANNELS]:
        raise ValueError(f"unexpected output view shape parameter: {view_shape!r}")

    device = conv.device
    if any(t.device != device for t in (cat, mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return conv, cat, mean_t, var_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([128, 16, 4, 4], f32), T([128, 168, 4, 4], f32), T([184], f32), T([184], f32), T([184], f32), T([184], f32), S([128, 184]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: accepts the same seven inputs as Repro.forward() and
    returns the same single f32 `[128,184]` contiguous tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu_mean.py")

    conv, cat, mean, var, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided((BATCH, CHANNELS), (CHANNELS, 1), device=conv.device, dtype=torch.float32)
    grid = (triton.cdiv(BATCH, BLOCK_N), triton.cdiv(CHANNELS, BLOCK_C))
    _cat_bn_relu_spatial_mean_kernel[grid](
        conv,
        cat,
        mean,
        var,
        weight,
        bias,
        out,
        BLOCK_BATCH=BLOCK_N,
        BLOCK_CHANNELS=BLOCK_C,
        BLOCK_ROWS=BLOCK_N * BLOCK_C,
        BLOCK_HW=SPATIAL,
        eps=EPS,
        num_warps=8,
        num_stages=3,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and expected.stride() == actual.stride()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    return ok


def _run_check(instance: torch.nn.Module, inputs: list[Any], *, atol: float, rtol: float) -> bool:
    """Correctness check that treats matching NaNs as equal.

    The repro's generic random variance input can contain negative values, so
    eager `sqrt(variance + eps)` legitimately produces NaNs. The shared harness
    interprets those NaNs as stochastic output drift; this local check verifies
    the exact NaN mask plus finite values for this deterministic repro.
    """
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False
    if expected.shape != actual.shape:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
            f"eager={list(expected.shape)}"
        )
        return False

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_mask_ok = torch.equal(expected_nan, actual_nan)
    finite = ~expected_nan
    if finite.any():
        max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        values_ok = torch.allclose(expected_f32[finite], actual_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = nan_mask_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} max_finite_diff={max_diff:.2e} "
        f"nan_count={int(expected_nan.sum().item())})"
    )
    if expected.dtype != actual.dtype:
        print(f"  output 0: WARNING dtype mismatch oracle={actual.dtype} eager={expected.dtype}")
    return ok


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
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
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
