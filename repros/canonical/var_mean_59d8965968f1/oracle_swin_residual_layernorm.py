"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin residual LayerNorm scope in one generated-code-matching Triton row-reduction kernel, including the `[6272,1024] -> [128,49,1024] -> [128,7,7,1024]` metadata views, residual-first add with `view_626`, fp32 population `var_mean(..., dim=3, correction=0, keepdim=True)`, `variance + 1e-5` before `libdevice.rsqrt`, affine scale/bias, singleton window metadata views, and final contiguous `[6272,1024]` output, whereas Inductor already emits one fused persistent reduction kernel for this full scope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a new narrow pattern because the remaining work is dominated by the required residual/addmm/affine reads, one 1024-wide row reduction, reciprocal square root, and output store; the fix is BANDWIDTH_BOUND: record this as an at-floor LayerNorm case unless broader normalization-template, launch-overhead, or memory-traffic improvements move both implementations."""
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
TOKENS = HEIGHT * WIDTH
HIDDEN = 1024
ROWS = BATCH * TOKENS
EPS = 1.0e-5
DTYPE = torch.float32

ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, TOKENS, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
VIEW_0_SHAPE = (BATCH, TOKENS, HIDDEN)
VIEW_1_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
WINDOW_VIEW_SHAPE = (BATCH, 1, HEIGHT, 1, WIDTH, HIDDEN)
WINDOW_FLAT_SHAPE = (-1, HEIGHT, WIDTH, HIDDEN)
ATTN_VIEW_SHAPE = (-1, TOKENS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


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
                "in_ptr3": "*fp32",
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
                    (6,): [["tt.divisibility", 16]],
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
            "num_load": 4,
            "num_store": 1,
            "num_reduction": 4,
            "autotune_hints": set(),
            "tiling_scores": {"x": 0, "r0_": 102768640},
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
        in_ptr3,
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
        tmp1 = tl.load(
            in_ptr1 + (r0_1 + 1024 * x0),
            xmask,
            eviction_policy="evict_first",
            other=0.0,
        )
        tmp25 = tl.load(in_ptr2 + r0_1, None, eviction_policy="evict_last")
        tmp27 = tl.load(in_ptr3 + r0_1, None, eviction_policy="evict_last")
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = tl.where(xmask, tmp3, 0)
        tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp8 = tl.where(xmask, tmp6, 0)
        tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
        tmp10 = tl.full([1, 1], 1024, tl.int32).to(tl.float32)
        tmp11 = tmp9 / tmp10
        tmp12 = tmp3 - tmp11
        tmp13 = tmp12 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = tl.where(xmask, tmp14, 0)
        tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
        tmp18 = tmp2 - tmp11
        tmp19 = tl.full([1, 1], 1024.0, tl.float32)
        tmp20 = tmp17 / tmp19
        tmp21 = tl.full([1, 1], 1e-05, tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp26 = tmp24 * tmp25
        tmp28 = tmp26 + tmp27
        tl.store(out_ptr2 + (r0_1 + 1024 * x0), tmp28, xmask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
    expected_stride: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected_shape}")
    if tuple(value.stride()) != expected_stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {expected_stride}")
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {expected_dtype}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_91", inputs[0], ADDMM_SHAPE, OUTPUT_STRIDE, DTYPE)
    residual = _require_tensor(
        "view_626",
        inputs[1],
        RESIDUAL_SHAPE,
        (TOKENS * HIDDEN, HIDDEN, 1),
        DTYPE,
    )
    weight = _require_tensor("arg347_1", inputs[2], AFFINE_SHAPE, (1,), DTYPE)
    bias = _require_tensor("arg348_1", inputs[3], AFFINE_SHAPE, (1,), DTYPE)

    shape0 = _shape_tuple(inputs[4])
    shape1 = _shape_tuple(inputs[5])
    shape2 = _shape_tuple(inputs[6])
    shape3 = _shape_tuple(inputs[7])
    shape4 = _shape_tuple(inputs[8])
    shape5 = _shape_tuple(inputs[9])

    expected_shapes = (
        VIEW_0_SHAPE,
        VIEW_1_SHAPE,
        WINDOW_VIEW_SHAPE,
        WINDOW_FLAT_SHAPE,
        ATTN_VIEW_SHAPE,
        OUTPUT_SHAPE,
    )
    actual_shapes = (shape0, shape1, shape2, shape3, shape4, shape5)
    for index, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes), start=4):
        if actual != expected:
            raise ValueError(f"shape parameter {index} is {actual}, expected {expected}")

    device = addmm.device
    for name, tensor in (
        ("view_626", residual),
        ("arg347_1", weight),
        ("arg348_1", bias),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return addmm, residual, weight, bias, shape0, shape1, shape2, shape3, shape4, shape5


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias, shape0, shape1, shape2, shape3, shape4, shape5 = (
        _validate_inputs(inputs)
    )
    x = torch.ops.aten.reshape.default(addmm, shape0)
    x = torch.ops.aten.add.Tensor(residual, x)
    x = torch.ops.aten.reshape.default(x, shape1)
    variance, mean = torch.ops.aten.var_mean.correction(
        x,
        [3],
        correction=0,
        keepdim=True,
    )
    centered = torch.ops.aten.sub.Tensor(x, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    x = torch.ops.aten.mul.Tensor(centered, invstd)
    x = torch.ops.aten.mul.Tensor(x, weight)
    x = torch.ops.aten.add.Tensor(x, bias)
    x = torch.ops.aten.reshape.default(x, shape2)
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.reshape.default(x, shape3)
    x = torch.ops.aten.reshape.default(x, shape4)
    return torch.ops.aten.reshape.default(x, shape5)


@oracle_impl(hardware="H100", shapes="(T([6272, 1024], f32), T([128, 49, 1024], f32), T([1024], f32), T([1024], f32), S([128, 49, 1024]), S([128, 7, 7, 1024]), S([128, 1, 7, 1, 7, 1024]), S([-1, 7, 7, 1024]), S([-1, 49, 1024]), S([6272, 1024]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward computation.

    SCOPE INVARIANT: accepts the same 10 inputs as Repro.forward() and returns
    the same single float32 contiguous `[6272, 1024]` output tensor.
    """
    addmm, residual, weight, bias, _shape0, _shape1, _shape2, _shape3, _shape4, shape5 = (
        _validate_inputs(inputs)
    )
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        shape5,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=DTYPE,
    )
    device_index = addmm.device.index if addmm.device.index is not None else torch.cuda.current_device()
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        triton_per_fused_add_mul_rsqrt_sub_var_mean_view_0.run(
            residual,
            addmm,
            weight,
            bias,
            output,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return output


def _layout_signature(value: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...], tuple[int, ...], int]:
    return value.dtype, tuple(value.shape), tuple(value.stride()), int(value.storage_offset())


def _check_layout_and_alias(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if isinstance(actual, torch.Tensor) and actual.is_cuda:
            torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    layout_ok = _layout_signature(expected) == _layout_signature(actual)
    alias_ok = expected.data_ptr() != inputs[0].data_ptr() and actual.data_ptr() != inputs[0].data_ptr()
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(expected={_layout_signature(expected)} oracle={_layout_signature(actual)})"
    )
    print(
        f"  output aliasing: {'PASS' if alias_ok else 'FAIL'} "
        "(single output, fresh storage)"
    )
    return layout_ok and alias_ok


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
        ok = _check_layout_and_alias(instance, inputs) and ok
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
