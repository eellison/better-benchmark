"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete four-scale BN-affine, NaN-preserving ReLU, bilinear-index resize, and final channel cat with branch-specialized Triton kernels that write the scale-1 branch directly into the contiguous [1, 256, 256, 256] output and materialize only the smaller reused BN/ReLU resize sources, whereas Inductor already emits comparable materialized BN/ReLU producers plus resize kernels that write into the final cat layout; Inductor cannot materially improve this local repro through additional scheduler fusion because the remaining time is dominated by required output stores, reused resize-source materialization, and bilinear stencil reads; the fix is BANDWIDTH_BOUND: record this full-scope layout/indexing stencil case as at floor unless broader memory-traffic or resize codegen improvements move both paths."""
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
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)

CLASSIFICATION = "BANDWIDTH_BOUND"
CHANNELS = 64
OUT_CHANNELS = 256
OUT_H = 256
OUT_W = 256
OUT_SHAPE = (1, OUT_CHANNELS, OUT_H, OUT_W)
OUT_STRIDE = (OUT_CHANNELS * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where(x > 0.0, x, tl.where(x < 0.0, 0.0, x))

    @triton.jit
    def _nan_like(x):
        return x + float("nan")

    @triton.jit
    def _bn_relu_value(conv_value, mean, var, weight, bias, valid_var):
        safe_var = tl.where(valid_var, var + 0.00001, 1.0)
        inv_std = 1.0 / tl.sqrt(safe_var)
        affine = ((conv_value - mean) * inv_std) * weight + bias
        return _relu_preserve_nan(affine)

    @triton.jit
    def _bn_relu_kernel(
        mean,
        conv,
        var,
        weight,
        bias,
        out_tmp,
        xnumel: tl.constexpr,
        HW: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = offsets < xnumel
        channel = offsets // HW

        mean_v = tl.load(mean + channel, mask=mask, other=0.0, eviction_policy="evict_last")
        var_v = tl.load(var + channel, mask=mask, other=0.0, eviction_policy="evict_last")
        weight_v = tl.load(weight + channel, mask=mask, other=0.0, eviction_policy="evict_last")
        bias_v = tl.load(bias + channel, mask=mask, other=0.0, eviction_policy="evict_last")
        valid_var = mask & ((var_v + 0.00001) >= 0.0)
        conv_v = tl.load(conv + offsets, mask=valid_var, other=0.0)
        value = _bn_relu_value(conv_v, mean_v, var_v, weight_v, bias_v, valid_var)
        value = tl.where(valid_var, value, _nan_like(value))
        tl.store(out_tmp + offsets, value, mask=mask)

    @triton.jit
    def _resize_to_output_kernel(
        src,
        var,
        out,
        xnumel: tl.constexpr,
        IN_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
        OUT_OFFSET: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = offsets < xnumel

        out_w = offsets % 256
        out_h = (offsets // 256) % 256
        channel = offsets // 65536

        var_v = tl.load(var + channel, mask=mask, other=0.0, eviction_policy="evict_last")
        valid_var = mask & ((var_v + 0.00001) >= 0.0)

        src_h_f = out_h.to(tl.float32) * SCALE
        src_h_f = tl.maximum(src_h_f, 0.0)
        h_lo = src_h_f.to(tl.int64)
        h_hi = tl.minimum(h_lo + 1, IN_SIZE - 1)
        h_frac = src_h_f - h_lo.to(tl.float32)
        h_frac = tl.minimum(tl.maximum(h_frac, 0.0), 1.0)

        src_w_f = out_w.to(tl.float32) * SCALE
        src_w_f = tl.maximum(src_w_f, 0.0)
        w_lo = src_w_f.to(tl.int64)
        w_hi = tl.minimum(w_lo + 1, IN_SIZE - 1)
        w_frac = src_w_f - w_lo.to(tl.float32)
        w_frac = tl.minimum(tl.maximum(w_frac, 0.0), 1.0)

        base = channel * (IN_SIZE * IN_SIZE)
        v_hi_lo = tl.load(src + base + h_hi * IN_SIZE + w_lo, mask=valid_var, other=0.0)
        v_hi_hi = tl.load(src + base + h_hi * IN_SIZE + w_hi, mask=valid_var, other=0.0)
        v_lo_lo = tl.load(src + base + h_lo * IN_SIZE + w_lo, mask=valid_var, other=0.0)
        v_lo_hi = tl.load(src + base + h_lo * IN_SIZE + w_hi, mask=valid_var, other=0.0)

        interp_hi = v_hi_lo + (v_hi_hi - v_hi_lo) * w_frac
        interp_lo = v_lo_lo + (v_lo_hi - v_lo_lo) * w_frac
        value = interp_lo + (interp_hi - interp_lo) * h_frac
        value = tl.where(valid_var, value, _nan_like(value))
        tl.store(out + OUT_OFFSET + offsets, value, mask=mask)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _expect_tensor(
    value: Any,
    name: str,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    device: torch.device | None,
) -> torch.device:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} expected shape {shape}, got {tuple(value.shape)}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} expected stride {stride}, got {tuple(value.stride())}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} expected torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if device is not None and value.device != device:
        raise ValueError(f"{name} is on {value.device}, expected {device}")
    return value.device


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 24:
        raise ValueError(f"{REPRO_ID} expects 24 inputs, got {len(inputs)}")

    device: torch.device | None = None
    tensor_specs = (
        ("arg287_1", (64,), (1,)),
        ("convolution_57", (1, 64, 256, 256), (4194304, 65536, 256, 1)),
        ("arg288_1", (64,), (1,)),
        ("arg289_1", (64,), (1,)),
        ("arg290_1", (64,), (1,)),
        ("arg292_1", (64,), (1,)),
        ("convolution_58", (1, 64, 128, 128), (1048576, 16384, 128, 1)),
        ("arg293_1", (64,), (1,)),
        ("arg294_1", (64,), (1,)),
        ("arg295_1", (64,), (1,)),
        ("arg297_1", (64,), (1,)),
        ("convolution_59", (1, 64, 64, 64), (262144, 4096, 64, 1)),
        ("arg298_1", (64,), (1,)),
        ("arg299_1", (64,), (1,)),
        ("arg300_1", (64,), (1,)),
        ("arg302_1", (64,), (1,)),
        ("convolution_60", (1, 64, 32, 32), (65536, 1024, 32, 1)),
        ("arg303_1", (64,), (1,)),
        ("arg304_1", (64,), (1,)),
        ("arg305_1", (64,), (1,)),
    )
    for index, (name, shape, stride) in enumerate(tensor_specs):
        device = _expect_tensor(inputs[index], name, shape, stride, device)

    shape_params = tuple(_shape_tuple(inputs[index], f"_shape_param_{index - 20}") for index in range(20, 24))
    if shape_params != ((256, 1), (256, 1), (256, 1), (256, 1)):
        raise ValueError(f"unexpected shape parameters: {shape_params}")

    return tuple(inputs[:20])


@oracle_impl(hardware="H100", shapes="(T([64], f32), T([1, 64, 256, 256], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([1, 64, 128, 128], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([1, 64, 64, 64], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([1, 64, 32, 32], f32), T([64], f32), T([64], f32), T([64], f32), S([256, 1]), S([256, 1]), S([256, 1]), S([256, 1]))")
def oracle_forward(inputs):
    """Run the full four-branch BN/ReLU/bilinear-resize/channel-cat scope."""
    tensors = _validate_inputs(inputs)
    (
        mean0,
        conv0,
        var0,
        weight0,
        bias0,
        mean1,
        conv1,
        var1,
        weight1,
        bias1,
        mean2,
        conv2,
        var2,
        weight2,
        bias2,
        mean3,
        conv3,
        var3,
        weight3,
        bias3,
    ) = tensors

    out = torch.empty(OUT_SHAPE, device=tensors[1].device, dtype=torch.float32)
    tmp1 = torch.empty((1, 64, 128, 128), device=out.device, dtype=torch.float32)
    tmp2 = torch.empty((1, 64, 64, 64), device=out.device, dtype=torch.float32)
    tmp3 = torch.empty((1, 64, 32, 32), device=out.device, dtype=torch.float32)

    _bn_relu_kernel[(triton.cdiv(4194304, 1024),)](
        mean0,
        conv0,
        var0,
        weight0,
        bias0,
        out,
        xnumel=4194304,
        HW=65536,
        XBLOCK=1024,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_kernel[(triton.cdiv(1048576, 1024),)](
        mean1,
        conv1,
        var1,
        weight1,
        bias1,
        tmp1,
        xnumel=1048576,
        HW=16384,
        XBLOCK=1024,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_kernel[(triton.cdiv(262144, 1024),)](
        mean2,
        conv2,
        var2,
        weight2,
        bias2,
        tmp2,
        xnumel=262144,
        HW=4096,
        XBLOCK=1024,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_kernel[(triton.cdiv(65536, 1024),)](
        mean3,
        conv3,
        var3,
        weight3,
        bias3,
        tmp3,
        xnumel=65536,
        HW=1024,
        XBLOCK=1024,
        num_warps=4,
        num_stages=3,
    )

    _resize_to_output_kernel[(triton.cdiv(4194304, 256),)](
        tmp1,
        var1,
        out,
        xnumel=4194304,
        IN_SIZE=128,
        SCALE=0.4980392156862745,
        OUT_OFFSET=4194304,
        XBLOCK=256,
        num_warps=4,
        num_stages=3,
    )
    _resize_to_output_kernel[(triton.cdiv(4194304, 256),)](
        tmp2,
        var2,
        out,
        xnumel=4194304,
        IN_SIZE=64,
        SCALE=0.24705882352941178,
        OUT_OFFSET=8388608,
        XBLOCK=256,
        num_warps=4,
        num_stages=3,
    )
    _resize_to_output_kernel[(triton.cdiv(4194304, 256),)](
        tmp3,
        var3,
        out,
        xnumel=4194304,
        IN_SIZE=32,
        SCALE=0.12156862745098039,
        OUT_OFFSET=12582912,
        XBLOCK=256,
        num_warps=4,
        num_stages=3,
    )
    return out


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_oracle_nan_aware(
    oracle_fn,
    instance,
    inputs,
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Check deterministic values while treating matching NaN masks as correct."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_fn(inputs)

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(eager_list) != len(oracle_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_ok = True
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        shape_ok = expected.shape == actual.shape
        dtype_ok = expected.dtype == actual.dtype
        stride_ok = expected.stride() == actual.stride()
        if not shape_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
                f"eager={list(expected.shape)}"
            )
            all_ok = False
            continue

        if not expected.is_floating_point():
            values_ok = torch.equal(expected, actual)
            max_diff = 0.0 if values_ok else float("inf")
            nan_ok = True
        else:
            expected_f32 = expected.float()
            actual_f32 = actual.float()
            expected_nan = torch.isnan(expected_f32)
            actual_nan = torch.isnan(actual_f32)
            nan_ok = torch.equal(expected_nan, actual_nan)
            finite = ~(expected_nan | actual_nan)
            if finite.any():
                finite_expected = expected_f32[finite]
                finite_actual = actual_f32[finite]
                values_ok = torch.allclose(finite_expected, finite_actual, atol=atol, rtol=rtol)
                max_diff = (finite_expected - finite_actual).abs().max().item()
            else:
                values_ok = True
                max_diff = 0.0

        ok = shape_ok and dtype_ok and stride_ok and nan_ok and values_ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  output {i}: {status} (shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={actual.stride()} max_diff={max_diff:.2e} "
            f"nan_mask={'PASS' if nan_ok else 'FAIL'})"
        )
        if not dtype_ok:
            print(f"    dtype mismatch: oracle={actual.dtype} eager={expected.dtype}")
        if not stride_ok:
            print(f"    stride mismatch: oracle={actual.stride()} eager={expected.stride()}")
        if not values_ok:
            print("    finite values differ")
        all_ok = all_ok and ok

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
        ok = _check_oracle_nan_aware(
            oracle_forward,
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
