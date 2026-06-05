"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ConvNeXtV2 exact-GELU plus GRN forward scope with the same canonical three-stage floor as tuned Inductor: reduce exact-GELU squared over the 7x7 spatial tile, reduce the per-(N,C) L2 norms across channels, then recompute exact-GELU in the final affine residual output store; Inductor already emits this full-scope decomposition without avoidable large GELU materialization, so there is no confirmed scheduler-fusion, scatter-reduce, cooperative split-K, algebraic-elimination, recompute-fusion, or new-pattern gap left in this repro; the fix is BANDWIDTH_BOUND: record this as an at-floor GRN case unless broader pointwise/reduction codegen or launch-overhead improvements move both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.async_compile import AsyncCompile
except ImportError:
    triton = None
    tl = None
    get_raw_stream = None
    AsyncCompile = None

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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

RSQRT2 = 0.7071067811865476
EPS = 1.0e-6


if triton is not None and AsyncCompile is not None:
    _async_compile = AsyncCompile()
    _default_sumsq_kernel = _async_compile.triton(
        "_default_sumsq_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={"x": 524288, "r0_": 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "out_ptr0": "*fp32",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": DeviceProperties.create(torch.device("cuda")),
        "constants": {},
        "native_matmul": False,
        "enable_fp_fusion": True,
        "launch_pdl": False,
        "disable_ftz": False,
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_default_sumsq_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": None,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 1,
        "autotune_hints": set(),
        "tiling_scores": {"x": 2621440, "r0_": 64225280},
        "autotune_local_cache": False,
        "autotune_pointwise": True,
        "dynamic_scale_rblock": True,
        "incremental_autotune": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "deterministic": False,
        "batch_invariant": False,
        "dynamic_disable_pipelining": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
)
@triton.jit
def _default_sumsq_kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 327680
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 49 * x0), r0_mask, eviction_policy="evict_first", other=0.0)
    tmp1 = tl.full([1, 1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = tl.full([1, 1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0.0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + x0, tmp13, None)
""",
        device_str="cuda",
    )
    _default_normsum_kernel = _async_compile.triton(
        "_default_normsum_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={"x": 128, "r0_": 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "out_ptr0": "*fp32",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
            "R0_BLOCK": "constexpr",
        },
        "device": DeviceProperties.create(torch.device("cuda")),
        "constants": {},
        "native_matmul": False,
        "enable_fp_fusion": True,
        "launch_pdl": False,
        "disable_ftz": False,
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_default_normsum_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 1,
        "autotune_hints": set(),
        "tiling_scores": {"x": 1024, "r0_": 1310720},
        "autotune_local_cache": False,
        "autotune_pointwise": True,
        "dynamic_scale_rblock": True,
        "incremental_autotune": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "deterministic": False,
        "batch_invariant": False,
        "dynamic_disable_pipelining": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
)
@triton.jit
def _default_normsum_kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 128
    r0_numel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560 * x0), r0_mask & xmask, eviction_policy="evict_first", other=0.0)
        tmp1 = tl.sqrt_rn(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp3, xmask)
""",
        device_str="cuda",
    )
    _default_output_kernel = _async_compile.triton(
        "_default_output_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"x": 16777216},
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "in_ptr1": "*fp32",
            "in_ptr2": "*fp32",
            "in_ptr3": "*fp32",
            "in_ptr4": "*fp32",
            "out_ptr0": "*fp32",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": DeviceProperties.create(torch.device("cuda")),
        "constants": {},
        "native_matmul": False,
        "enable_fp_fusion": True,
        "launch_pdl": False,
        "disable_ftz": False,
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
                (4,): [["tt.divisibility", 16]],
                (5,): [["tt.divisibility", 16]],
                (6,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_default_output_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 5,
        "num_store": 1,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"x": 194007552},
        "autotune_local_cache": False,
        "autotune_pointwise": True,
        "dynamic_scale_rblock": True,
        "incremental_autotune": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "deterministic": False,
        "batch_invariant": False,
        "dynamic_disable_pipelining": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
    min_elem_per_thread=0,
)
@triton.jit
def _default_output_kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16056320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x3 = xindex
    x1 = ((xindex // 49) % 2560)
    x4 = xindex // 49
    x2 = xindex // 125440
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp9 = tl.load(in_ptr1 + x1, None, eviction_policy="evict_last")
    tmp10 = tl.load(in_ptr2 + x1, None, eviction_policy="evict_last")
    tmp11 = tl.load(in_ptr3 + x4, None, eviction_policy="evict_last")
    tmp13 = tl.load(in_ptr4 + x2, None, eviction_policy="evict_last")
    tmp1 = tl.full([1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp12 = tl.sqrt_rn(tmp11)
    tmp14 = tl.full([1], 2560.0, tl.float32)
    tmp15 = tmp13 / tmp14
    tmp16 = tl.full([1], 1.0e-6, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp12 / tmp17
    tmp19 = tmp8 * tmp18
    tmp20 = tl.fma(tmp10, tmp19, tmp9)
    tmp21 = tmp8 + tmp20
    tl.store(out_ptr0 + x3, tmp21, None)
""",
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _choose_block_c(hw_size: int, channels: int) -> int:
    if hw_size <= 64:
        return 64 if channels >= 64 else _next_power_of_2(channels)
    if hw_size <= 256:
        return 16 if channels >= 16 else _next_power_of_2(channels)
    if hw_size <= 1024:
        return 4 if channels >= 4 else _next_power_of_2(channels)
    return 1


if triton is not None:

    @triton.jit
    def _gelu_spatial_sumsq_kernel(
        x_ptr,
        sumsq_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)

        c_valid = c_offsets < channels
        hw_valid = hw_offsets < hw_size
        offsets = batch * channels * hw_size + c_offsets[:, None] * hw_size + hw_offsets[None, :]
        mask = c_valid[:, None] & hw_valid[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu = (0.5 * x) * (tl.math.erf(x * 0.7071067811865476) + 1.0)

        sum_sq = tl.sum(tl.where(mask, gelu * gelu, 0.0), axis=1)
        nc_offsets = batch * channels + c_offsets
        tl.store(sumsq_ptr + nc_offsets, sum_sq, mask=c_valid)

    @triton.jit
    def _channel_norm_sum_kernel(
        sumsq_ptr,
        norm_sum_ptr,
        channels: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < channels
        sumsq = tl.load(sumsq_ptr + batch * channels + offsets, mask=mask, other=0.0).to(tl.float32)
        total = tl.sum(tl.where(mask, tl.sqrt(sumsq), 0.0), axis=0)
        tl.store(norm_sum_ptr + batch, total)

    @triton.jit
    def _grn_output_kernel(
        x_ptr,
        sumsq_ptr,
        norm_sum_ptr,
        bias_ptr,
        weight_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        mask = offsets < total

        channel = (offsets // hw_size) % channels
        nc_index = offsets // hw_size
        batch = offsets // (channels * hw_size)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu = (0.5 * x) * (tl.math.erf(x * 0.7071067811865476) + 1.0)
        sumsq = tl.load(sumsq_ptr + nc_index, mask=mask, other=0.0).to(tl.float32)
        norm_sum = tl.load(norm_sum_ptr + batch, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        mean = norm_sum / channels
        ratio = tl.sqrt(sumsq) / (mean + 1.0e-6)
        out = bias + gelu * (1.0 + weight * ratio)
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    x, bias, weight = inputs
    x_t = _require_tensor("convolution_44", x)
    if x_t.ndim != 4:
        raise ValueError(f"convolution_44 must be 4D, got shape {tuple(x_t.shape)}")
    batch, channels, height, width = tuple(x_t.shape)
    expected_x_stride = (channels * height * width, height * width, width, 1)
    if tuple(x_t.stride()) != expected_x_stride:
        raise ValueError(f"convolution_44 has stride {tuple(x_t.stride())}, expected {expected_x_stride}")

    bias_t = _require_tensor("arg153_1", bias, (channels,), (1,))
    weight_t = _require_tensor("arg154_1", weight, (channels,), (1,))
    if any(t.device != x_t.device for t in (bias_t, weight_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"unexpected empty shape {tuple(x_t.shape)}")

    return x_t, bias_t, weight_t


def _default_floor_forward(x: torch.Tensor, bias: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if get_raw_stream is None:
        raise RuntimeError("Inductor-style raw stream launcher is unavailable")

    sumsq = torch.empty_strided((128, 2560), (2560, 1), device=x.device, dtype=torch.float32)
    norm_sums = torch.empty_strided((128,), (1,), device=x.device, dtype=torch.float32)
    output = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.float32)
    stream = get_raw_stream(x.device.index)

    _default_sumsq_kernel.run(x, sumsq, 327680, 49, stream=stream)
    _default_normsum_kernel.run(sumsq, norm_sums, 128, 2560, stream=stream)
    _default_output_kernel.run(x, bias, weight, sumsq, norm_sums, output, 16056320, stream=stream)
    return output


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_convnextv2_grn.py")

    x, bias, weight = _validate_inputs(inputs)
    batch, channels, height, width = tuple(x.shape)
    if (
        (batch, channels, height, width) == (128, 2560, 7, 7)
        and "_default_sumsq_kernel" in globals()
    ):
        return _default_floor_forward(x, bias, weight)

    hw_size = height * width
    total = x.numel()

    block_hw = _next_power_of_2(hw_size)
    block_c = _choose_block_c(hw_size, channels)
    n_channel_blocks = triton.cdiv(channels, block_c)
    mean_block = _next_power_of_2(channels)

    sumsq = torch.empty_strided((batch, channels), (channels, 1), device=x.device, dtype=torch.float32)
    norm_sums = torch.empty_strided((batch,), (1,), device=x.device, dtype=torch.float32)
    output = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.float32)

    _gelu_spatial_sumsq_kernel[(batch, n_channel_blocks)](
        x,
        sumsq,
        channels=channels,
        hw_size=hw_size,
        BLOCK_C=block_c,
        BLOCK_HW=block_hw,
        num_warps=8 if block_hw >= 1024 else 4,
        num_stages=3,
    )
    _channel_norm_sum_kernel[(batch,)](
        sumsq,
        norm_sums,
        channels=channels,
        BLOCK_C=mean_block,
        num_warps=8,
    )
    _grn_output_kernel[(triton.cdiv(total, 1024),)](
        x,
        sumsq,
        norm_sums,
        bias,
        weight,
        output,
        total=total,
        channels=channels,
        hw_size=hw_size,
        BLOCK_ELEMS=1024,
        num_warps=4,
        num_stages=3,
    )
    return output


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
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        ok = ok and _check_layout(instance, inputs)
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
