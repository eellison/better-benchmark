"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete BN-inference affine broadcast in one storage-order Triton pointwise kernel using an rsqrt-normalized affine expression, whereas Inductor emits a generic flattened pointwise kernel for the decomposed unsqueeze/sub/sqrt/reciprocal/mul/add graph that keeps the sqrt followed by reciprocal and broadcast affine chain in elementwise form; Inductor cannot do this today because its algebraic simplifier and pointwise codegen do not canonicalize BN-inference variance normalization to rsqrt affine arithmetic before lowering the broadcast, even when the output layout is a simple contiguous or channels-last NCHW storage walk; the fix is ALGEBRAIC_ELIMINATION: add a guarded BN-affine canonicalization that rewrites sqrt-plus-reciprocal normalization to indexed rsqrt affine code for both contiguous and channels-last NCHW layouts."""
from __future__ import annotations

import argparse
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

EPS = 1.0e-5
LAYOUT_CONTIGUOUS = 0
LAYOUT_CHANNELS_LAST = 1
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL", "C", "HW", "LAYOUT"],
    )
    @triton.jit
    def _bn_affine_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        LAYOUT: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL

        if LAYOUT == 0:
            c_offsets = (offsets // HW) % C
        else:
            c_offsets = offsets % C

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * tl.rsqrt(var + eps) * weight + bias
        tl.store(out_ptr + offsets, y, mask=mask)


def _require_f32_cuda_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0")
    return value


def _layout_kind(x: torch.Tensor) -> int:
    if x.is_contiguous():
        return LAYOUT_CONTIGUOUS
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last):
        return LAYOUT_CHANNELS_LAST
    raise ValueError(
        f"unsupported activation layout for {REPRO_ID}: "
        f"shape={tuple(x.shape)} stride={tuple(x.stride())}"
    )


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_affine.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    mean, x, var, weight, bias = (
        _require_f32_cuda_tensor(name, value)
        for name, value in zip(
            ("arg415_1", "convolution_91", "arg416_1", "arg417_1", "arg418_1"),
            inputs,
        )
    )

    if x.dim() != 4:
        raise ValueError(f"convolution_91 must be rank 4, got shape={tuple(x.shape)}")

    batch, channels, height, width = x.shape
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"convolution_91 has invalid shape {tuple(x.shape)}")

    for name, tensor in (
        ("arg415_1", mean),
        ("arg416_1", var),
        ("arg417_1", weight),
        ("arg418_1", bias),
    ):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"{name} shape must be ({channels},), got {tuple(tensor.shape)}")
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same CUDA device as convolution_91")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    return mean, x, var, weight, bias, _layout_kind(x)


@oracle_impl(hardware="H100", shapes="(T([80], f32), T([512, 80, 7, 7], f32), T([80], f32), T([80], f32), T([80], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with folded BN-inference affine coefficients."""
    mean, x, var, weight, bias, layout = _validate_inputs(inputs)
    _, channels, height, width = x.shape
    hw = height * width
    total = x.numel()

    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_N"]),)
    _bn_affine_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        TOTAL=total,
        C=channels,
        HW=hw,
        LAYOUT=layout,
        eps=EPS,
    )
    return output


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (tuple, list)):
        return tuple(outputs)
    return (outputs,)


def _check_oracle_nan_aware(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    all_pass = True
    for i, (expected_value, actual_value) in enumerate(zip(expected_list, actual_list)):
        if not isinstance(expected_value, torch.Tensor) or not isinstance(actual_value, torch.Tensor):
            ok = expected_value == actual_value
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        metadata_ok = (
            expected_value.shape == actual_value.shape
            and expected_value.dtype == actual_value.dtype
            and expected_value.stride() == actual_value.stride()
            and expected_value.device == actual_value.device
        )
        if not metadata_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"oracle=(shape={list(actual_value.shape)} stride={actual_value.stride()} "
                f"dtype={actual_value.dtype}) "
                f"eager=(shape={list(expected_value.shape)} stride={expected_value.stride()} "
                f"dtype={expected_value.dtype})"
            )
            all_pass = False
            continue

        if not expected_value.is_floating_point():
            ok = torch.equal(expected_value, actual_value)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected_value.dtype})")
            all_pass = all_pass and ok
            continue

        expected_f32 = expected_value.float()
        actual_f32 = actual_value.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_mask_ok = torch.equal(expected_nan, actual_nan)
        finite = ~(expected_nan | actual_nan)
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
            finite_ok = torch.allclose(expected_f32[finite], actual_f32[finite], atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            finite_ok = True

        ok = nan_mask_ok and finite_ok
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_value.shape)} dtype={expected_value.dtype} "
            f"stride={expected_value.stride()} max_finite_diff={max_diff:.2e} "
            f"nan_count={int(expected_nan.sum().item())})"
        )
        all_pass = all_pass and ok

    return all_pass


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

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_oracle_nan_aware(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
