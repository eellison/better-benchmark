"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BEiT class-token plus patch-token LayerNorm scope with the same generated Inductor Welford schedule, including expand/reshape/permute/cat semantics, population `var_mean(..., correction=0)`, `eps` before `libdevice.rsqrt`, affine scale/bias, and the final contiguous `[25216, 768]` reshape, whereas Inductor already emits the equivalent one-kernel full-scope reduction; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new pattern because the remaining cost is the required token/patch reads, Welford statistics, affine reads, rsqrt, and output store; the fix is BANDWIDTH_BOUND: record this as an at-floor BEiT patch-token LayerNorm case unless broader normalization-template or memory-traffic work moves both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.async_compile import AsyncCompile
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
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


BATCH = 128
CHANNELS = 768
PATCH_H = 14
PATCH_W = 14
PATCHES = PATCH_H * PATCH_W
TOKENS = PATCHES + 1
ROWS = BATCH * TOKENS
EPS = 1.0e-6
DTYPE = torch.float32

CLASS_TOKEN_SHAPE = (1, 1, CHANNELS)
CLASS_TOKEN_STRIDE = (CHANNELS, CHANNELS, 1)
CONV_SHAPE = (BATCH, CHANNELS, PATCH_H, PATCH_W)
CONV_STRIDE = (CHANNELS * PATCHES, 1, PATCH_W * CHANNELS, CHANNELS)
AFFINE_SHAPE = (CHANNELS,)
AFFINE_STRIDE = (1,)
EXPAND_SHAPE = (BATCH, -1, -1)
CONV_RESHAPE = (BATCH, CHANNELS, PATCHES)
OUTPUT_SHAPE = (ROWS, CHANNELS)
AFFINE_OUTPUT_SHAPE = (BATCH, TOKENS, CHANNELS)
AFFINE_OUTPUT_STRIDE = (TOKENS * CHANNELS, CHANNELS, 1)
OUTPUT_STRIDE = (CHANNELS, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


def _shape_tuple(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"{name} must be a shape sequence, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {DTYPE}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    class_token = _require_tensor("arg3_1", inputs[0], CLASS_TOKEN_SHAPE, CLASS_TOKEN_STRIDE)
    convolution = _require_tensor("convolution", inputs[1], CONV_SHAPE, CONV_STRIDE)
    weight = _require_tensor("arg5_1", inputs[2], AFFINE_SHAPE, AFFINE_STRIDE)
    bias = _require_tensor("arg6_1", inputs[3], AFFINE_SHAPE, AFFINE_STRIDE)

    device = class_token.device
    for name, tensor in (
        ("convolution", convolution),
        ("arg5_1", weight),
        ("arg6_1", bias),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} device {tensor.device} != {device}")

    expand_shape = _shape_tuple("_shape_param_0", inputs[4])
    conv_reshape = _shape_tuple("_shape_param_1", inputs[5])
    output_shape = _shape_tuple("_shape_param_2", inputs[6])
    if expand_shape != EXPAND_SHAPE:
        raise ValueError(f"unexpected class-token expand shape: {expand_shape!r}")
    if conv_reshape != CONV_RESHAPE:
        raise ValueError(f"unexpected convolution reshape shape: {conv_reshape!r}")
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output reshape shape: {output_shape!r}")

    return class_token, convolution, weight, bias, output_shape


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    (
        class_token,
        convolution,
        weight,
        bias,
        expand_shape,
        conv_reshape,
        output_shape,
    ) = inputs
    expanded = torch.ops.aten.expand.default(class_token, expand_shape)
    reshaped = torch.ops.aten.reshape.default(convolution, conv_reshape)
    patches = torch.ops.aten.permute.default(reshaped, [0, 2, 1])
    cat = torch.ops.aten.cat.default([expanded, patches], 1)
    variance, mean = torch.ops.aten.var_mean.correction(
        cat,
        [2],
        correction=0,
        keepdim=True,
    )
    centered = torch.ops.aten.sub.Tensor(cat, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    scaled = torch.ops.aten.mul.Tensor(normalized, weight)
    biased = torch.ops.aten.add.Tensor(scaled, bias)
    return torch.ops.aten.reshape.default(biased, output_shape)


# --- Oracle kernel(s) ---
_inductor_floor_kernel = None
if triton is not None and AsyncCompile is not None:
    _async_compile = AsyncCompile()
    _inductor_floor_kernel = _async_compile.triton(
        "triton_red_fused_add_cat_expand_mul_permute_rsqrt_sub_var_mean_view_0",
        r'''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_add_cat_expand_mul_permute_rsqrt_sub_var_mean_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 232006656}, 'backend_hash': '2FF7A49C450C045FF2AC7A730B5A8D4CD6DF40914BC38271CC70D96733C19B52', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_add_cat_expand_mul_permute_rsqrt_sub_var_mean_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 25216
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 197)
    x1 = xindex // 197
    tmp14_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp14_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp14_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp0 >= tmp5
        tmp9 = tl.full([1, 1], 197, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + (r0_2 + 768*((-1) + x0) + 150528*x1), r0_mask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.where(tmp6, tmp7, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp14_mean_next, tmp14_m2_next, tmp14_weight_next = triton_helpers.welford_reduce(
            tmp13, tmp14_mean, tmp14_m2, tmp14_weight, roffset == 0
        )
        tmp14_mean = tl.where(r0_mask & xmask, tmp14_mean_next, tmp14_mean)
        tmp14_m2 = tl.where(r0_mask & xmask, tmp14_m2_next, tmp14_m2)
        tmp14_weight = tl.where(r0_mask & xmask, tmp14_weight_next, tmp14_weight)
    tmp15, tmp16, tmp17 = triton_helpers.welford(tmp14_mean, tmp14_m2, tmp14_weight, 1)
    tmp14 = tmp15[:, None]
    tmp18 = tmp16[:, None]
    tmp19 = tmp17[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp40 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp42 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp20 = (x0).to(tl.int32)
        tmp21 = tl.full([1, 1], 0, tl.int64)
        tmp22 = tmp20 >= tmp21
        tmp23 = (x0).to(tl.int64)
        tmp24 = (tmp23).to(tl.int64)
        tmp25 = tl.full([1, 1], 1, tl.int64)
        tmp26 = tmp24 < tmp25
        tmp27 = tl.load(in_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp26 & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp20 >= tmp25
        tmp29 = tl.full([1, 1], 197, tl.int64)
        tmp30 = tmp20 < tmp29
        tmp31 = tl.load(in_ptr1 + (r0_2 + 768*((-1) + x0) + 150528*x1), r0_mask & tmp28 & xmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.where(tmp26, tmp27, tmp31)
        tmp33 = tmp32 - tmp14
        tmp34 = tl.full([1, 1], 768.0, tl.float32)
        tmp35 = (tmp18 / tmp34)
        tmp36 = tl.full([1, 1], 1e-06, tl.float32)
        tmp37 = tmp35 + tmp36
        tmp38 = libdevice.rsqrt(tmp37)
        tmp39 = tmp33 * tmp38
        tmp41 = tmp39 * tmp40
        tmp43 = tmp41 + tmp42
        tl.store(out_ptr2 + (r0_2 + 768*x3), tmp43, r0_mask & xmask)
''',
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope and return the flattened output."""
    class_token, convolution, weight, bias, output_shape = _validate_inputs(inputs)
    if (
        triton is None
        or _inductor_floor_kernel is None
        or get_raw_stream is None
        or not convolution.is_cuda
    ):
        return _torch_full_scope(inputs)

    affine_out = torch.empty_strided(
        AFFINE_OUTPUT_SHAPE,
        AFFINE_OUTPUT_STRIDE,
        device=convolution.device,
        dtype=DTYPE,
    )
    device_index = convolution.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _inductor_floor_kernel.run(
        class_token,
        convolution,
        weight,
        bias,
        affine_out,
        ROWS,
        CHANNELS,
        stream=get_raw_stream(device_index),
    )
    return torch.ops.aten.reshape.default(affine_out, output_shape)


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()
    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == OUTPUT_SHAPE
        and tuple(actual.stride()) == tuple(expected.stride()) == OUTPUT_STRIDE
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    return ok


# --- CLI entry point ---
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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
