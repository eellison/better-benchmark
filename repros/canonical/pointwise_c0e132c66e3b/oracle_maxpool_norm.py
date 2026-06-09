"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BN-inference affine in fp32, fp16 cast, ReLU, and 3x3 stride-2 ceil-mode low-memory maxpool-with-offsets in one shape-specialized Triton kernel that writes only the final fp16 values and int8 offsets, whereas Inductor currently materializes the full fp16 ReLU activation before a separate multi-output pooling stencil; Inductor cannot do this today because scheduler fusion does not sink a pointwise normalized producer through prims low-memory maxpool-with-offsets while preserving offset semantics; the fix is SCHEDULER_FUSION: allow affine/ReLU producers to be inlined into low-memory maxpool-with-offsets and emit values plus offsets from the same loop nest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)

BATCH = 32
CHANNELS = 768
HEIGHT = 14
WIDTH = 14
OUT_HEIGHT = 7
OUT_WIDTH = 7
HW = HEIGHT * WIDTH
OUT_HW = OUT_HEIGHT * OUT_WIDTH
EPS = 1.0e-5
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUTPUT_SHAPE = (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
OUTPUT_STRIDE = (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1)
BLOCK_C = 4
BLOCK_OUT = 64


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_relu_maxpool_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        values_ptr,
        offsets_ptr,
        BLOCK_C_: tl.constexpr,
        BLOCK_OUT_: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)

        c_offsets = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        out_offsets = tl.arange(0, BLOCK_OUT_)
        out_h = out_offsets // 7
        out_w = out_offsets - out_h * 7

        c_mask = c_offsets < 768
        out_mask = out_offsets < 49

        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(var + 1.0e-5)

        best = tl.full((BLOCK_C_, BLOCK_OUT_), -float("inf"), tl.float32)
        best_offset = tl.zeros((BLOCK_C_, BLOCK_OUT_), tl.int32)
        base = (n * 768 + c_offsets[:, None]) * 196

        for kh in tl.static_range(0, 3):
            in_h = out_h * 2 + kh
            valid_h = in_h < 14
            for kw in tl.static_range(0, 3):
                in_w = out_w * 2 + kw
                valid = c_mask[:, None] & out_mask[None, :] & valid_h[None, :] & (in_w[None, :] < 14)
                x = tl.load(
                    conv_ptr + base + in_h[None, :] * 14 + in_w[None, :],
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)

                affine = (x - mean[:, None]) * inv_std[:, None] * weight[:, None] + bias[:, None]
                affine_fp16 = affine.to(tl.float16)
                relu = tl.where(
                    affine_fp16 <= 0.0,
                    0.0,
                    affine_fp16.to(tl.float32),
                )

                better = valid & ((relu > best) | (relu != relu))
                best = tl.where(better, relu, best)
                best_offset = tl.where(better, kh * 3 + kw, best_offset)

        flat_out = (n * 768 + c_offsets[:, None]) * 49 + out_offsets[None, :]
        store_mask = c_mask[:, None] & out_mask[None, :]
        tl.store(values_ptr + flat_out, best, mask=store_mask)
        tl.store(offsets_ptr + flat_out, best_offset.to(tl.int8), mask=store_mask)


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
    if value.dtype is not torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_maxpool_norm.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    mean = _require_tensor("arg132_1", inputs[0], (CHANNELS,), (1,))
    conv = _require_tensor("convolution_26", inputs[1], INPUT_SHAPE, INPUT_STRIDE)
    var = _require_tensor("arg133_1", inputs[2], (CHANNELS,), (1,))
    weight = _require_tensor("arg134_1", inputs[3], (CHANNELS,), (1,))
    bias = _require_tensor("arg135_1", inputs[4], (CHANNELS,), (1,))

    device = conv.device
    if any(tensor.device != device for tensor in (mean, var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean, conv, var, weight, bias


@oracle_impl(hardware="H100", shapes="(T([768], f16), T([32, 768, 14, 14], f16), T([768], f16), T([768], f16), T([768], f16))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full BN-inference affine, fp16 ReLU, and low-memory maxpool scope."""
    mean, conv, var, weight, bias = _validate_inputs(inputs)
    values = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=conv.device,
        dtype=torch.float16,
    )
    offsets = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=conv.device,
        dtype=torch.int8,
    )

    grid = (BATCH, triton.cdiv(CHANNELS, BLOCK_C))
    _bn_relu_maxpool_kernel[grid](
        mean,
        conv,
        var,
        weight,
        bias,
        values,
        offsets,
        BLOCK_C_=BLOCK_C,
        BLOCK_OUT_=BLOCK_OUT,
        num_warps=8,
        num_stages=3,
    )
    return values, offsets


def _check_oracle_nan_equal(
    instance: torch.nn.Module,
    inputs: list[Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Correctness check for deterministic NaNs from unconstrained variance input."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(eager, tuple) or not isinstance(oracle_out, tuple):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tuples")
        return False
    if len(eager) != 2 or len(oracle_out) != 2:
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_out)} outputs, "
            f"eager produces {len(eager)}"
        )
        return False

    expected_values, expected_offsets = eager
    actual_values, actual_offsets = oracle_out
    all_ok = True

    if expected_values.shape != actual_values.shape:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(actual_values.shape)} "
            f"eager={list(expected_values.shape)}"
        )
        all_ok = False
    elif expected_values.dtype != actual_values.dtype:
        print(
            f"  output 0: SCOPE_MISMATCH dtype oracle={actual_values.dtype} "
            f"eager={expected_values.dtype}"
        )
        all_ok = False
    elif expected_values.stride() != actual_values.stride():
        print(
            f"  output 0: SCOPE_MISMATCH stride oracle={actual_values.stride()} "
            f"eager={expected_values.stride()}"
        )
        all_ok = False
    else:
        expected_f32 = expected_values.float()
        actual_f32 = actual_values.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_ok = torch.equal(expected_nan, actual_nan)
        finite = ~(expected_nan | actual_nan)
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
            values_ok = torch.allclose(expected_f32[finite], actual_f32[finite], atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            values_ok = True
        ok = nan_ok and values_ok
        print(
            f"  output 0: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_values.shape)} dtype={expected_values.dtype} "
            f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
        )
        all_ok = all_ok and ok

    if expected_offsets.shape != actual_offsets.shape:
        print(
            f"  output 1: SCOPE_MISMATCH shape oracle={list(actual_offsets.shape)} "
            f"eager={list(expected_offsets.shape)}"
        )
        all_ok = False
    elif expected_offsets.dtype != actual_offsets.dtype:
        print(
            f"  output 1: SCOPE_MISMATCH dtype oracle={actual_offsets.dtype} "
            f"eager={expected_offsets.dtype}"
        )
        all_ok = False
    elif expected_offsets.stride() != actual_offsets.stride():
        print(
            f"  output 1: SCOPE_MISMATCH stride oracle={actual_offsets.stride()} "
            f"eager={expected_offsets.stride()}"
        )
        all_ok = False
    else:
        offsets_ok = torch.equal(expected_offsets, actual_offsets)
        mismatch_count = int((expected_offsets != actual_offsets).sum().item())
        print(
            f"  output 1: {'PASS' if offsets_ok else 'FAIL'} "
            f"(exact, dtype={expected_offsets.dtype}, mismatch_count={mismatch_count})"
        )
        all_ok = all_ok and offsets_ok

    return all_ok


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_oracle_nan_equal(instance, inputs, atol=args.atol, rtol=args.rtol)
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
