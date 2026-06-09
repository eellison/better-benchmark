"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full six-branch Inception BN-inference affine, NaN-preserving ReLU, fixed channel concatenations, 8x8 spatial mean, and final contiguous f32 [128,2048] reshape by reducing each branch tensor directly into its output channel slice, whereas Inductor currently schedules the six BN/ReLU branch producers, two 768-channel inner cats, the 2048-channel outer cat, and the spatial mean as generic producer/consumer regions with materialized concat-shaped intermediates; Inductor cannot do this today because its scheduler does not keep fixed channel-cat producers virtual across a downstream reduction that needs per-output-channel source selection; the fix is SCHEDULER_FUSION: represent fixed-shape aten.cat as a virtual multi-source layout and sink the per-branch BN/ReLU pointwise epilogue into the fused spatial-mean reduction that writes the final reshape."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


BATCH = 128
C0 = 320
C1 = 384
C2 = 384
C3 = 384
C4 = 384
C5 = 192
CHANNELS = C0 + C1 + C2 + C3 + C4 + C5
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
EPS = 1.0e-3
BLOCK_BATCH = 1
BLOCK_CHANNELS = 16
BLOCK_ROWS = BLOCK_BATCH * BLOCK_CHANNELS
BLOCK_HW = 64
AFFINE_BLOCK = 256
CLASSIFICATION = "SCHEDULER_FUSION"
TRUE_FLOOR = True
ACTIONABLE = True


if triton is not None:

    @triton.jit
    def _affine_precompute_kernel(
        mean0_ptr,
        var0_ptr,
        weight0_ptr,
        bias0_ptr,
        mean1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        mean2_ptr,
        var2_ptr,
        weight2_ptr,
        bias2_ptr,
        mean3_ptr,
        var3_ptr,
        weight3_ptr,
        bias3_ptr,
        mean4_ptr,
        var4_ptr,
        weight4_ptr,
        bias4_ptr,
        mean5_ptr,
        var5_ptr,
        weight5_ptr,
        bias5_ptr,
        affine_ptr,
        BLOCK_CHANNELS_: tl.constexpr,
        eps: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        valid = offsets < 2048

        in0 = offsets < 320
        c0 = offsets
        in1 = (offsets >= 320) & (offsets < 704)
        c1 = offsets - 320
        in2 = (offsets >= 704) & (offsets < 1088)
        c2 = offsets - 704
        in3 = (offsets >= 1088) & (offsets < 1472)
        c3 = offsets - 1088
        in4 = (offsets >= 1472) & (offsets < 1856)
        c4 = offsets - 1472
        in5 = offsets >= 1856
        c5 = offsets - 1856

        mean = tl.full((BLOCK_CHANNELS_,), 0.0, tl.float32)
        var = tl.full((BLOCK_CHANNELS_,), 1.0, tl.float32)
        weight = tl.full((BLOCK_CHANNELS_,), 0.0, tl.float32)
        bias = tl.full((BLOCK_CHANNELS_,), 0.0, tl.float32)

        mean = tl.where(in0, tl.load(mean0_ptr + c0, mask=valid & in0, other=0.0), mean)
        var = tl.where(in0, tl.load(var0_ptr + c0, mask=valid & in0, other=1.0), var)
        weight = tl.where(in0, tl.load(weight0_ptr + c0, mask=valid & in0, other=0.0), weight)
        bias = tl.where(in0, tl.load(bias0_ptr + c0, mask=valid & in0, other=0.0), bias)

        mean = tl.where(in1, tl.load(mean1_ptr + c1, mask=valid & in1, other=0.0), mean)
        var = tl.where(in1, tl.load(var1_ptr + c1, mask=valid & in1, other=1.0), var)
        weight = tl.where(in1, tl.load(weight1_ptr + c1, mask=valid & in1, other=0.0), weight)
        bias = tl.where(in1, tl.load(bias1_ptr + c1, mask=valid & in1, other=0.0), bias)

        mean = tl.where(in2, tl.load(mean2_ptr + c2, mask=valid & in2, other=0.0), mean)
        var = tl.where(in2, tl.load(var2_ptr + c2, mask=valid & in2, other=1.0), var)
        weight = tl.where(in2, tl.load(weight2_ptr + c2, mask=valid & in2, other=0.0), weight)
        bias = tl.where(in2, tl.load(bias2_ptr + c2, mask=valid & in2, other=0.0), bias)

        mean = tl.where(in3, tl.load(mean3_ptr + c3, mask=valid & in3, other=0.0), mean)
        var = tl.where(in3, tl.load(var3_ptr + c3, mask=valid & in3, other=1.0), var)
        weight = tl.where(in3, tl.load(weight3_ptr + c3, mask=valid & in3, other=0.0), weight)
        bias = tl.where(in3, tl.load(bias3_ptr + c3, mask=valid & in3, other=0.0), bias)

        mean = tl.where(in4, tl.load(mean4_ptr + c4, mask=valid & in4, other=0.0), mean)
        var = tl.where(in4, tl.load(var4_ptr + c4, mask=valid & in4, other=1.0), var)
        weight = tl.where(in4, tl.load(weight4_ptr + c4, mask=valid & in4, other=0.0), weight)
        bias = tl.where(in4, tl.load(bias4_ptr + c4, mask=valid & in4, other=0.0), bias)

        mean = tl.where(in5, tl.load(mean5_ptr + c5, mask=valid & in5, other=0.0), mean)
        var = tl.where(in5, tl.load(var5_ptr + c5, mask=valid & in5, other=1.0), var)
        weight = tl.where(in5, tl.load(weight5_ptr + c5, mask=valid & in5, other=0.0), weight)
        bias = tl.where(in5, tl.load(bias5_ptr + c5, mask=valid & in5, other=0.0), bias)

        scale = (1.0 / tl.sqrt(var + eps)) * weight
        shift = bias - mean * scale
        tl.store(affine_ptr + offsets, scale, mask=valid)
        tl.store(affine_ptr + 2048 + offsets, shift, mask=valid)

    @triton.jit
    def _bn_relu_spatial_mean_branch_kernel(
        x_ptr,
        affine_ptr,
        out_ptr,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        channels: tl.constexpr,
        out_start: tl.constexpr,
        BLOCK_BATCH_: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        batch_block = tl.program_id(0)
        channel_block = tl.program_id(1)

        row_offsets = tl.arange(0, BLOCK_ROWS_)
        n_offsets = batch_block * BLOCK_BATCH_ + row_offsets // BLOCK_CHANNELS_
        c_offsets = channel_block * BLOCK_CHANNELS_ + row_offsets % BLOCK_CHANNELS_
        hw_offsets = tl.arange(0, BLOCK_HW_)
        h_offsets = hw_offsets // 8
        w_offsets = hw_offsets - h_offsets * 8

        valid_rows = (n_offsets < 128) & (c_offsets < channels)
        x_offsets = (
            n_offsets[:, None] * stride_n
            + c_offsets[:, None] * stride_c
            + h_offsets[None, :] * stride_h
            + w_offsets[None, :] * stride_w
        )
        x = tl.load(x_ptr + x_offsets, mask=valid_rows[:, None], other=0.0).to(tl.float32)

        global_c = out_start + c_offsets
        scale = tl.load(affine_ptr + global_c, mask=valid_rows, other=0.0).to(tl.float32)
        shift = tl.load(affine_ptr + 2048 + global_c, mask=valid_rows, other=0.0).to(tl.float32)
        y = x * scale[:, None] + shift[:, None]
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        reduced = tl.sum(tl.where(valid_rows[:, None], y, 0.0), axis=1) * 0.015625

        out_offsets = n_offsets * 2048 + global_c
        tl.store(out_ptr + out_offsets, reduced, mask=valid_rows)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    valid_strides: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) not in valid_strides:
        raise ValueError(
            f"{name} has stride {tuple(value.stride())}, expected one of {valid_strides}"
        )
    return value


def _valid_nchw_strides(channels: int) -> tuple[tuple[int, ...], ...]:
    return (
        (channels * HW, HW, WIDTH, 1),
        (channels * HW, 1, WIDTH * channels, channels),
    )


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 31:
        raise ValueError(f"{REPRO_ID} expects 31 inputs, got {len(inputs)}")

    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        mean3,
        x3,
        var3,
        weight3,
        bias3,
        mean4,
        x4,
        var4,
        weight4,
        bias4,
        mean5,
        x5,
        var5,
        weight5,
        bias5,
        view_shape,
    ) = inputs

    mean0_t = _require_f32_tensor("arg427_1", mean0, (C0,), ((1,),))
    x0_t = _require_f32_tensor(
        "convolution_85",
        x0,
        (BATCH, C0, HEIGHT, WIDTH),
        _valid_nchw_strides(C0),
    )
    var0_t = _require_f32_tensor("arg428_1", var0, (C0,), ((1,),))
    weight0_t = _require_f32_tensor("arg429_1", weight0, (C0,), ((1,),))
    bias0_t = _require_f32_tensor("arg430_1", bias0, (C0,), ((1,),))

    mean1_t = _require_f32_tensor("arg437_1", mean1, (C1,), ((1,),))
    x1_t = _require_f32_tensor(
        "convolution_87",
        x1,
        (BATCH, C1, HEIGHT, WIDTH),
        _valid_nchw_strides(C1),
    )
    var1_t = _require_f32_tensor("arg438_1", var1, (C1,), ((1,),))
    weight1_t = _require_f32_tensor("arg439_1", weight1, (C1,), ((1,),))
    bias1_t = _require_f32_tensor("arg440_1", bias1, (C1,), ((1,),))

    mean2_t = _require_f32_tensor("arg442_1", mean2, (C2,), ((1,),))
    x2_t = _require_f32_tensor(
        "convolution_88",
        x2,
        (BATCH, C2, HEIGHT, WIDTH),
        _valid_nchw_strides(C2),
    )
    var2_t = _require_f32_tensor("arg443_1", var2, (C2,), ((1,),))
    weight2_t = _require_f32_tensor("arg444_1", weight2, (C2,), ((1,),))
    bias2_t = _require_f32_tensor("arg445_1", bias2, (C2,), ((1,),))

    mean3_t = _require_f32_tensor("arg457_1", mean3, (C3,), ((1,),))
    x3_t = _require_f32_tensor(
        "convolution_91",
        x3,
        (BATCH, C3, HEIGHT, WIDTH),
        _valid_nchw_strides(C3),
    )
    var3_t = _require_f32_tensor("arg458_1", var3, (C3,), ((1,),))
    weight3_t = _require_f32_tensor("arg459_1", weight3, (C3,), ((1,),))
    bias3_t = _require_f32_tensor("arg460_1", bias3, (C3,), ((1,),))

    mean4_t = _require_f32_tensor("arg462_1", mean4, (C4,), ((1,),))
    x4_t = _require_f32_tensor(
        "convolution_92",
        x4,
        (BATCH, C4, HEIGHT, WIDTH),
        _valid_nchw_strides(C4),
    )
    var4_t = _require_f32_tensor("arg463_1", var4, (C4,), ((1,),))
    weight4_t = _require_f32_tensor("arg464_1", weight4, (C4,), ((1,),))
    bias4_t = _require_f32_tensor("arg465_1", bias4, (C4,), ((1,),))

    mean5_t = _require_f32_tensor("arg467_1", mean5, (C5,), ((1,),))
    x5_t = _require_f32_tensor(
        "convolution_93",
        x5,
        (BATCH, C5, HEIGHT, WIDTH),
        _valid_nchw_strides(C5),
    )
    var5_t = _require_f32_tensor("arg468_1", var5, (C5,), ((1,),))
    weight5_t = _require_f32_tensor("arg469_1", weight5, (C5,), ((1,),))
    bias5_t = _require_f32_tensor("arg470_1", bias5, (C5,), ((1,),))

    if list(view_shape) != [BATCH, CHANNELS]:
        raise ValueError(f"unexpected output reshape parameter: {view_shape!r}")

    tensors = (
        mean0_t,
        x0_t,
        var0_t,
        weight0_t,
        bias0_t,
        mean1_t,
        x1_t,
        var1_t,
        weight1_t,
        bias1_t,
        mean2_t,
        x2_t,
        var2_t,
        weight2_t,
        bias2_t,
        mean3_t,
        x3_t,
        var3_t,
        weight3_t,
        bias3_t,
        mean4_t,
        x4_t,
        var4_t,
        weight4_t,
        bias4_t,
        mean5_t,
        x5_t,
        var5_t,
        weight5_t,
        bias5_t,
    )
    device = x0_t.device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return tensors


def _launch_branch(
    x: torch.Tensor,
    affine: torch.Tensor,
    output: torch.Tensor,
    channels: int,
    out_start: int,
) -> None:
    grid = (triton.cdiv(BATCH, BLOCK_BATCH), triton.cdiv(channels, BLOCK_CHANNELS))
    _bn_relu_spatial_mean_branch_kernel[grid](
        x,
        affine,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        channels=channels,
        out_start=out_start,
        BLOCK_BATCH_=BLOCK_BATCH,
        BLOCK_CHANNELS_=BLOCK_CHANNELS,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )


@oracle_impl(hardware="H100", shapes="(T([320], f32), T([128, 320, 8, 8], f32, stride=(20480, 1, 2560, 320)), T([320], f32), T([320], f32), T([320], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([384], f32), T([192], f32), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([192], f32), T([192], f32), T([192], f32), S([128, 2048]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full captured Repro.forward computation.

    SCOPE INVARIANT: accepts the same 31 inputs as Repro.forward() and returns
    the same single f32 [128, 2048] contiguous tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_multi_bn_relu_spatial_mean.py")

    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        mean3,
        x3,
        var3,
        weight3,
        bias3,
        mean4,
        x4,
        var4,
        weight4,
        bias4,
        mean5,
        x5,
        var5,
        weight5,
        bias5,
    ) = _validate_inputs(inputs)

    affine = torch.empty_strided((2, CHANNELS), (CHANNELS, 1), device=x0.device, dtype=torch.float32)
    output = torch.empty_strided((BATCH, CHANNELS), (CHANNELS, 1), device=x0.device, dtype=torch.float32)

    _affine_precompute_kernel[(triton.cdiv(CHANNELS, AFFINE_BLOCK),)](
        mean0,
        var0,
        weight0,
        bias0,
        mean1,
        var1,
        weight1,
        bias1,
        mean2,
        var2,
        weight2,
        bias2,
        mean3,
        var3,
        weight3,
        bias3,
        mean4,
        var4,
        weight4,
        bias4,
        mean5,
        var5,
        weight5,
        bias5,
        affine,
        BLOCK_CHANNELS_=AFFINE_BLOCK,
        eps=EPS,
        num_warps=4,
        num_stages=3,
    )

    _launch_branch(x0, affine, output, C0, 0)
    _launch_branch(x1, affine, output, C1, C0)
    _launch_branch(x2, affine, output, C2, C0 + C1)
    _launch_branch(x3, affine, output, C3, C0 + C1 + C2)
    _launch_branch(x4, affine, output, C4, C0 + C1 + C2 + C3)
    _launch_branch(x5, affine, output, C5, C0 + C1 + C2 + C3 + C4)
    return output


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate values, dtype, shape, stride, and deterministic NaN masks."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    shape_ok = expected.shape == actual.shape
    dtype_ok = expected.dtype == actual.dtype
    stride_ok = expected.stride() == actual.stride()
    if not shape_ok:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
            f"eager={list(expected.shape)}"
        )
        return False

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_ok = torch.equal(expected_nan, actual_nan)
    finite = ~expected_nan & ~actual_nan
    if finite.any():
        max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        values_ok = torch.allclose(expected_f32[finite], actual_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    print(
        f"  output 0 values: {'PASS' if values_ok and dtype_ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} "
        f"oracle_dtype={actual.dtype} max_finite_diff={max_diff:.2e})"
    )
    print(
        f"  output 0 layout: {'PASS' if stride_ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    print(
        f"  output 0 NaNs: {'PASS' if nan_ok else 'FAIL'} "
        f"(expected_nan={int(expected_nan.sum().item())}, "
        f"oracle_nan={int(actual_nan.sum().item())})"
    )
    return values_ok and dtype_ok and stride_ok and nan_ok


# --- CLI entry point ---
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
