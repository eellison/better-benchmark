"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full captured MobileNetV3 training-BatchNorm hard-swish block, including per-channel var_mean over `[512, 960, 7, 7]`, both running-stat copy_ side effects, affine hard-swish, and the final keepdim spatial mean, using one split statistics kernel followed by one fused update/activation/pooling Triton kernel that preserves the returned running-stat aliases, whereas tuned Inductor reaches the same full-scope timing envelope for this single-shape NCHW normalization template; Inductor cannot materially improve this row through a local fusion change because the measured cost is dominated by the required statistics pass, activation/pooling pass, output stores, running-stat writes, and launch overhead already matched by the compiler; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope oracle and only revisit if a broader normalization-template or memory-traffic change moves the family."""
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
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    triton_helpers = None
    triton_heuristics = None
    DeviceProperties = None
    ReductionHint = None
    libdevice = None
    get_raw_stream = None

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


BATCH = 512
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS, 1, 1)
OUTPUT_STRIDE = (CHANNELS, 1, 1, 1)

if triton is not None:
    triton_helpers.set_driver_to_gpu()
    _DEVICE_PROPS = DeviceProperties(
        type="cuda",
        index=0,
        multi_processor_count=132,
        cc=90,
        major=9,
        regs_per_multiprocessor=65536,
        max_threads_per_multi_processor=2048,
        max_threads_per_block=1024,
        warp_size=32,
    )

    @triton_heuristics.reduction(
        size_hints={"x": 1024, "r0_": 32768},
        reduction_hint=ReductionHint.INNER,
        filename="/tmp/oracle_var_mean_mean_60c37a09d40e.py",
        triton_meta={
            "signature": {
                "in_ptr0": "*fp32",
                "in_ptr1": "*fp32",
                "in_ptr2": "*fp32",
                "out_ptr0": "*fp32",
                "out_ptr1": "*fp32",
                "out_ptr3": "*fp32",
                "out_ptr5": "*fp32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
                "R0_BLOCK": "constexpr",
            },
            "device": _DEVICE_PROPS,
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
                    (7,): [["tt.divisibility", 16]],
                    (8,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "_inductor_stats_update_kernel",
            "mutated_arg_names": ["in_ptr1", "in_ptr2", "out_ptr3", "out_ptr5"],
            "optimize_mem": True,
            "no_x_dim": False,
            "atomic_add_found": False,
            "num_load": 3,
            "num_store": 4,
            "num_reduction": 2,
            "autotune_hints": set(),
            "tiling_scores": {"x": 38400, "r0_": 96337920},
            "backend_hash": "2FF7A49C450C045FF2AC7A730B5A8D4CD6DF40914BC38271CC70D96733C19B52",
            "assert_indirect_indexing": True,
            "autotune_local_cache": True,
            "autotune_pointwise": True,
            "autotune_remote_cache": None,
            "force_disable_caches": False,
            "dynamic_scale_rblock": True,
            "incremental_autotune": False,
            "max_autotune": False,
            "max_autotune_pointwise": False,
            "min_split_scan_rblock": 256,
            "spill_threshold": 16,
            "store_cubin": False,
            "deterministic": False,
            "batch_invariant": False,
            "force_filter_reduction_configs": False,
            "mix_order_reduction_allow_multi_stages": True,
            "dynamic_disable_pipelining": True,
            "are_deterministic_algorithms_enabled": False,
            "coordinate_descent_tuning": True,
            "coordinate_descent_search_radius": 1,
            "coordinate_descent_check_all_directions": False,
        },
    )
    @triton.jit
    def _inductor_stats_update_kernel(
        in_ptr0,
        in_ptr1,
        in_ptr2,
        out_ptr0,
        out_ptr1,
        out_ptr3,
        out_ptr5,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        R0_BLOCK: tl.constexpr,
    ):
        xnumel = 960
        r0_numel = 25088
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex
        tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index % 49
            r0_2 = r0_index // 49
            tmp0 = tl.load(
                in_ptr0 + (r0_1 + 49 * x0 + 47040 * r0_2),
                r0_mask & xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
            tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
                tmp1, tmp2_mean, tmp2_m2, tmp2_weight, r0_offset == 0
            )
            tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
            tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
            tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
        tmp3, tmp4, _ = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
        mean = tmp3[:, None]
        m2 = tmp4[:, None]
        tl.store(out_ptr0 + x0, mean, xmask)
        tl.store(out_ptr1 + x0, m2, xmask)
        old_mean = tl.load(in_ptr1 + x0, xmask)
        old_var = tl.load(in_ptr2 + x0, xmask)
        updated_mean = mean * 0.1 + old_mean * 0.9
        updated_var = (m2 / 25088.0) * 1.0000398612827361 * 0.1 + old_var * 0.9
        tl.store(out_ptr3 + x0, updated_mean, xmask)
        tl.store(out_ptr5 + x0, updated_var, xmask)

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 524288, "r0_": 64},
        reduction_hint=ReductionHint.INNER,
        filename="/tmp/oracle_var_mean_mean_60c37a09d40e.py",
        triton_meta={
            "signature": {
                "in_out_ptr0": "*fp32",
                "in_ptr0": "*fp32",
                "in_ptr1": "*fp32",
                "in_ptr2": "*fp32",
                "in_ptr3": "*fp32",
                "in_ptr4": "*fp32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
            },
            "device": _DEVICE_PROPS,
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
            "kernel_name": "_inductor_hardswish_spatial_mean_kernel",
            "mutated_arg_names": ["in_out_ptr0"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 5,
            "num_store": 1,
            "num_reduction": 1,
            "autotune_hints": set(),
            "tiling_scores": {"x": 3947520, "r0_": 96337920},
            "backend_hash": "2FF7A49C450C045FF2AC7A730B5A8D4CD6DF40914BC38271CC70D96733C19B52",
            "assert_indirect_indexing": True,
            "autotune_local_cache": True,
            "autotune_pointwise": True,
            "autotune_remote_cache": None,
            "force_disable_caches": False,
            "dynamic_scale_rblock": True,
            "incremental_autotune": False,
            "max_autotune": False,
            "max_autotune_pointwise": False,
            "min_split_scan_rblock": 256,
            "spill_threshold": 16,
            "store_cubin": False,
            "deterministic": False,
            "batch_invariant": False,
            "force_filter_reduction_configs": False,
            "mix_order_reduction_allow_multi_stages": True,
            "dynamic_disable_pipelining": True,
            "are_deterministic_algorithms_enabled": False,
            "coordinate_descent_tuning": True,
            "coordinate_descent_search_radius": 1,
            "coordinate_descent_check_all_directions": False,
        },
    )
    @triton.jit
    def _inductor_hardswish_spatial_mean_kernel(
        in_out_ptr0,
        in_ptr0,
        in_ptr1,
        in_ptr2,
        in_ptr3,
        in_ptr4,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 491520
        r0_numel = 49
        R0_BLOCK: tl.constexpr = 64
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = r0_index < r0_numel
        x3 = xindex
        x0 = xindex % 960
        tmp0 = tl.load(in_ptr0 + (r0_index + 49 * x3), r0_mask, eviction_policy="evict_first", other=0.0)
        tmp1 = tl.load(in_ptr1 + x0, None, eviction_policy="evict_last")
        tmp3 = tl.load(in_ptr2 + x0, None, eviction_policy="evict_last")
        tmp10 = tl.load(in_ptr3 + x0, None, eviction_policy="evict_last")
        tmp12 = tl.load(in_ptr4 + x0, None, eviction_policy="evict_last")
        tmp2 = tmp0 - tmp1
        tmp5 = tmp3 / 25088.0
        tmp8 = libdevice.rsqrt(tmp5 + 1.0e-5)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + 3.0
        tmp17 = triton_helpers.maximum(tmp15, 0.0)
        tmp19 = triton_helpers.minimum(tmp17, 6.0)
        tmp22 = (tmp13 * tmp19) * 0.16666666666666666
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = tl.where(r0_mask, tmp23, 0.0)
        tmp26 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
        tmp28 = tmp26 / 49.0
        tl.store(in_out_ptr0 + x3, tmp28, None)


def _expect_f32_tensor(
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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    convolution_57, arg289_1, arg290_1, arg291_1, arg292_1 = inputs
    x = _expect_f32_tensor("convolution_57", convolution_57, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg289_1", arg289_1, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var = _expect_f32_tensor("arg290_1", arg290_1, VECTOR_SHAPE, VECTOR_STRIDE)
    weight = _expect_f32_tensor("arg291_1", arg291_1, VECTOR_SHAPE, VECTOR_STRIDE)
    bias = _expect_f32_tensor("arg292_1", arg292_1, VECTOR_SHAPE, VECTOR_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


@oracle_impl(hardware="H100", shapes="(T([512, 960, 7, 7], f32), T([960], f32), T([960], f32), T([960], f32), T([960], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope, including running-stat copy_ effects."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_training_hardswish_spatial_mean.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    mean = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=x.device, dtype=torch.float32)
    m2 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=x.device, dtype=torch.float32)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    raw_stream = get_raw_stream(x.device.index or 0)
    _inductor_stats_update_kernel.run(
        x,
        running_mean,
        running_var,
        mean,
        m2,
        running_mean,
        running_var,
        CHANNELS,
        ELEMENTS_PER_CHANNEL,
        stream=raw_stream,
    )
    _inductor_hardswish_spatial_mean_kernel.run(
        output,
        x,
        mean,
        m2,
        weight,
        bias,
        BATCH * CHANNELS,
        HW,
        stream=raw_stream,
    )
    return output, running_mean, running_var


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
