"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin singleton-window LayerNorm scope in one fixed-hidden Triton row kernel, including the metadata-only `[6272,1024] -> [128,7,7,1024]` view, fp32 population var_mean over hidden size 1024, eps=1e-5 before `libdevice.rsqrt`, affine scale/bias, singleton window-partition view chain, and final contiguous `[6272,1024]` output, whereas Inductor already lowers this stacktrace-labeled variant as the same single normalization-template region with only metadata views around it; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a new narrow pattern because the remaining work is the mandatory activation/affine reads, one row reduction, rsqrt, and output store; the fix is BANDWIDTH_BOUND: record this full-scope LayerNorm case as at floor unless broader normalization-template, launch, or bandwidth work moves both paths."""
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


BATCH = 128
HEIGHT = 7
WIDTH = 7
HIDDEN = 1024
ROWS = BATCH * HEIGHT * WIDTH
INPUT_SHAPE = (ROWS, HIDDEN)
VIEW_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
WINDOW_VIEW_SHAPE = (BATCH, 1, HEIGHT, 1, WIDTH, HIDDEN)
WINDOW_FLAT_PARAM = (-1, HEIGHT, WIDTH, HIDDEN)
ATTN_VIEW_PARAM = (-1, HEIGHT * WIDTH, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-5
BLOCK_H = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects eight inputs, got {len(inputs)}")

    activation = _require_tensor("mm_2", inputs[0], INPUT_SHAPE, torch.float32)
    weight = _require_tensor("arg333_1", inputs[1], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg334_1", inputs[2], (HIDDEN,), torch.float32)

    expected_shapes = (
        VIEW_SHAPE,
        WINDOW_VIEW_SHAPE,
        WINDOW_FLAT_PARAM,
        ATTN_VIEW_PARAM,
        OUTPUT_SHAPE,
    )
    for index, expected in enumerate(expected_shapes, start=3):
        actual = _shape_tuple(inputs[index])
        if actual != expected:
            raise ValueError(f"shape parameter {index} is {actual}, expected {expected}")

    device = activation.device
    if weight.device != device or bias.device != device:
        raise ValueError("all tensor inputs must be on the same device")

    return activation, weight, bias, OUTPUT_SHAPE


# --- Oracle kernel(s) ---

if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 8192, "r0_": 1024},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "in_ptr0": "*fp32",
                "in_ptr1": "*fp32",
                "in_ptr2": "*fp32",
                "out_ptr2": "*fp32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
            },
            "device": DeviceProperties.create(torch.device("cuda", 0)),
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
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "triton_per_fused_add_mul_rsqrt_sub_var_mean_view_0",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 3,
            "num_store": 1,
            "num_reduction": 4,
            "autotune_hints": set(),
            "tiling_scores": {"x": 0, "r0_": 77078528},
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
        },
    )
    @triton.jit
    def triton_per_fused_add_mul_rsqrt_sub_var_mean_view_0(
        in_ptr0,
        in_ptr1,
        in_ptr2,
        out_ptr2,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 6272
        r0_numel = 1024
        R0_BLOCK: tl.constexpr = 1024
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_offset = 0
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp0 = tl.load(
            in_ptr0 + (r0_1 + 1024 * x0),
            xmask,
            eviction_policy="evict_first",
            other=0.0,
        )
        tmp23 = tl.load(in_ptr1 + r0_1, None, eviction_policy="evict_last")
        tmp25 = tl.load(in_ptr2 + r0_1, None, eviction_policy="evict_last")
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = tl.where(xmask, tmp1, 0)
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp6 = tl.where(xmask, tmp4, 0)
        tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
        tmp8 = tl.full([1, 1], 1024, tl.int32).to(tl.float32)
        tmp9 = tmp7 / tmp8
        tmp10 = tmp1 - tmp9
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = tl.where(xmask, tmp12, 0)
        tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
        tmp16 = tmp0 - tmp9
        tmp17 = tl.full([1, 1], 1024.0, tl.float32)
        tmp18 = tmp15 / tmp17
        tmp19 = tl.full([1, 1], 1e-05, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp22 * tmp23
        tmp26 = tmp24 + tmp25
        tl.store(out_ptr2 + (r0_1 + 1024 * x0), tmp26, xmask)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    activation, weight, bias, output_shape = _validate_inputs(inputs)
    x = torch.ops.aten.reshape.default(activation, _shape_tuple(inputs[3]))
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [3], correction=0, keepdim=True
    )
    centered = torch.ops.aten.sub.Tensor(x, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    x = torch.ops.aten.mul.Tensor(centered, invstd)
    x = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(x, weight), bias)
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[4]))
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[5]))
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[6]))
    return torch.ops.aten.reshape.default(x, output_shape)


@oracle_impl(hardware="H100", shapes="(T([6272, 1024], f32), T([1024], f32), T([1024], f32), S([128, 7, 7, 1024]), S([128, 1, 7, 1, 7, 1024]), S([-1, 7, 7, 1024]), S([-1, 49, 1024]), S([6272, 1024]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward singleton-window LayerNorm computation.

    SCOPE INVARIANT: accepts the same eight inputs as Repro.forward() and returns
    the same single fp32 contiguous `[6272,1024]` tensor.
    """
    activation, weight, bias, output_shape = _validate_inputs(inputs)
    if triton is None or not activation.is_cuda or not weight.is_cuda or not bias.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        output_shape,
        OUTPUT_STRIDE,
        device=activation.device,
        dtype=torch.float32,
    )
    device_index = (
        activation.device.index
        if activation.device.index is not None
        else torch.cuda.current_device()
    )
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        triton_per_fused_add_mul_rsqrt_sub_var_mean_view_0.run(
            activation,
            weight,
            bias,
            output,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return output


def _layout_signature(value: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...], tuple[int, ...]]:
    return value.dtype, tuple(value.shape), tuple(value.stride())


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    ok = _layout_signature(expected) == _layout_signature(actual)
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected={_layout_signature(expected)} oracle={_layout_signature(actual)})"
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
        ok = _check_layout(instance, inputs) and ok
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
