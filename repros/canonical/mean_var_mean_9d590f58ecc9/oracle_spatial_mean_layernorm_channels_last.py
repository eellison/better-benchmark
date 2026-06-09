"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ConvNeXtV2 channels-last spatial-mean plus channel-LayerNorm scope, including the input add, 7x7 mean, channel var_mean correction=0, affine epilogue, final [128,640] reshape, and rsqrt/640 side output, but forces the spatial-mean kernel onto the inner/coalesced persistent-reduction schedule for the captured NCHW shape with NHWC physical strides, whereas Inductor fuses the same full scope but classifies that strided 7x7 spatial reduction with the generic DEFAULT reduction schedule and lands on a much slower tiling; Inductor cannot do this today because its norm-template canonicalization/scheduler heuristics do not recognize this channels-last ConvNeXt spatial-mean-then-channel-LayerNorm layout as the same coalesced reduction template; the fix is NEW_PATTERN: add a channels-last spatial-mean LayerNorm canonicalization or reduction-hint rule that selects the inner persistent-reduction schedule without changing the semantic two-kernel scope."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    _async_compile = AsyncCompile()

    _spatial_mean_kernel = _async_compile.triton("oracle_9d590_spatial_mean", '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import ReductionHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_9d590_spatial_mean', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 655360, 'r0_': 32112640}, 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def oracle_9d590_spatial_mean(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 81920
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    n_index = xindex // 640
    c_index = xindex - (n_index * 640)
    input_offset = c_index + (31360 * n_index) + (640 * r0_index)
    tmp0 = tl.load(in_ptr0 + input_offset, r0_mask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.load(in_ptr1 + input_offset, r0_mask, eviction_policy='evict_first', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp3, 0.0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp6 = tmp5 / 49.0
    tl.store(in_out_ptr0 + xindex, tmp6, None)
''', device_str='cuda')

    _layernorm_kernel = _async_compile.triton("oracle_9d590_layernorm", '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import ReductionHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_9d590_layernorm', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 1024, 'r0_': 988160}, 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def oracle_9d590_layernorm(in_out_ptr0, in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 128
    r0_numel = 640
    R0_BLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    tmp0 = tl.load(in_out_ptr0 + (r0_index + 640 * xindex), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.load(in_ptr0 + r0_index, r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr1 + r0_index, r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp3, 0.0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp6 = tmp5 / 640.0
    tmp7 = tmp3 - tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp9, 0.0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tmp0 - tmp6
    tmp13 = tmp11 / 640.0
    tmp14 = tmp13 + 1.0e-6
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 + tmp2
    tmp19 = tmp15 * 0.0015625
    tl.store(in_out_ptr0 + (r0_index + 640 * xindex), tmp18, r0_mask & xmask)
    tl.store(out_ptr2 + xindex, tmp19, xmask)
''', device_str='cuda')

    _async_compile.wait(globals())
    del _async_compile


@oracle_impl(hardware="H100", shapes="(T([128, 640, 7, 7], f32, stride=(31360, 1, 4480, 640)), T([128, 640, 7, 7], f32, stride=(31360, 1, 4480, 640)), T([640], f32), T([640], f32), S([128, 640]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None or AsyncCompile is None or get_raw_stream is None:
        raise RuntimeError("Triton/Inductor runtime is required for this oracle")

    x0, x1, weight, bias, shape = inputs
    n, c, h, w = x0.shape
    if x0.dtype != torch.float32 or x1.dtype != torch.float32:
        raise TypeError("oracle expects float32 activation inputs")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("oracle expects float32 affine inputs")
    if x1.shape != x0.shape:
        raise ValueError("activation input shapes must match")
    if tuple(weight.shape) != (c,) or tuple(bias.shape) != (c,):
        raise ValueError(f"affine inputs must have shape {(c,)}")
    if tuple(shape) != (n, c):
        raise ValueError(f"shape parameter {shape} does not match activation shape {(n, c)}")
    if (n, c, h, w) != (128, 640, 7, 7):
        raise ValueError(f"oracle is specialized for (128, 640, 7, 7), got {(n, c, h, w)}")
    expected_stride = (31360, 1, 4480, 640)
    if tuple(x0.stride()) != expected_stride or tuple(x1.stride()) != expected_stride:
        raise ValueError(f"oracle expects channels-last NCHW stride {expected_stride}")

    inv_out = torch.empty_strided((n, 1, 1, 1), (1, 1, 1, 1), device=x0.device, dtype=x0.dtype)
    spatial_mean = torch.empty_strided((n, c), (c, 1), device=x0.device, dtype=x0.dtype)

    raw_stream = get_raw_stream(x0.device.index or 0)
    _spatial_mean_kernel.run(
        spatial_mean,
        x0,
        x1,
        n * c,
        h * w,
        stream=raw_stream,
    )
    _layernorm_kernel.run(
        spatial_mean,
        weight,
        bias,
        inv_out,
        n,
        c,
        stream=raw_stream,
    )
    return (spatial_mean, inv_out)


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
