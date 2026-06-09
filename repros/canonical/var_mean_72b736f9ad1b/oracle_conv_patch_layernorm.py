"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ViT convolution-patch affine LayerNorm scope as a standalone shape-specialized Welford row kernel, including the `[128,768,16,16] -> [128,768,256] -> [128,256,768]` metadata path, broadcast add, population `var_mean(correction=0)`, `libdevice.rsqrt(var + 1e-6)`, affine epilogue, contiguous clone materialization, and final `[32768,768]` unsafe view, whereas Inductor already emits the same fused two-pass Welford kernel for this full scope; Inductor cannot remove more local work while preserving exact Welford numerics because the row statistics must be completed before the affine output pass and the final output must be freshly materialized; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless broader normalization launch or memory-traffic changes move both implementations."""
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
BATCH = 128
CHANNELS = 768
PATCHES = 256
HEIGHT = 16
WIDTH = 16
ROWS = BATCH * PATCHES
HIDDEN = CHANNELS
EPS = 1.0e-6

CONV_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
CONV_STRIDE = (CHANNELS * PATCHES, PATCHES, WIDTH, 1)
PATCH_BIAS_SHAPE = (1, PATCHES, HIDDEN)
PATCH_BIAS_STRIDE = (PATCHES * HIDDEN, HIDDEN, 1)
AFFINE_SHAPE = (HIDDEN,)
AFFINE_STRIDE = (1,)
VIEW_SHAPE = (BATCH, CHANNELS, PATCHES)
OUTPUT_3D_SHAPE = (BATCH, PATCHES, HIDDEN)
OUTPUT_3D_STRIDE = (PATCHES * HIDDEN, HIDDEN, 1)
OUTPUT_FLAT_SHAPE = (ROWS, HIDDEN)
OUTPUT_FLAT_STRIDE = (HIDDEN, 1)


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

    @triton_heuristics.reduction(
        size_hints={"x": 32768, "r0_": 1024},
        reduction_hint=ReductionHint.DEFAULT,
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
                "R0_BLOCK": "constexpr",
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
            "kernel_name": "oracle_conv_patch_layernorm_kernel",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": False,
            "atomic_add_found": False,
            "num_load": 6,
            "num_store": 1,
            "num_reduction": 2,
            "autotune_hints": set(),
            "tiling_scores": {"x": 100663296, "r0_": 202119168},
            "backend_hash": "oracle",
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
    def _conv_patch_layernorm_kernel(
        in_ptr0,
        in_ptr1,
        in_ptr2,
        in_ptr3,
        out_ptr2,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        R0_BLOCK: tl.constexpr,
    ):
        xnumel = 32768
        r0_numel = 768
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = xindex % 256
        x1 = xindex // 256
        tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(
                in_ptr0 + (x0 + 256 * r0_2 + 196608 * x1),
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp1 = tl.load(
                in_ptr1 + (r0_2 + 768 * x0),
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp2 = tmp0 + tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
                tmp3,
                tmp4_mean,
                tmp4_m2,
                tmp4_weight,
                roffset == 0,
            )
            tmp4_mean = tl.where(r0_mask, tmp4_mean_next, tmp4_mean)
            tmp4_m2 = tl.where(r0_mask, tmp4_m2_next, tmp4_m2)
            tmp4_weight = tl.where(r0_mask, tmp4_weight_next, tmp4_weight)
        tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
        tmp4 = tmp5[:, None]
        tmp8 = tmp6[:, None]
        tmp9 = tmp7[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp10 = tl.load(
                in_ptr0 + (x0 + 256 * r0_2 + 196608 * x1),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp11 = tl.load(
                in_ptr1 + (r0_2 + 768 * x0),
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp20 = tl.load(
                in_ptr2 + r0_2,
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp22 = tl.load(
                in_ptr3 + r0_2,
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp12 = tmp10 + tmp11
            tmp13 = tmp12 - tmp4
            tmp14 = tl.full([1, 1], 768.0, tl.float32)
            tmp15 = tmp8 / tmp14
            tmp16 = tl.full([1, 1], 1.0e-6, tl.float32)
            tmp17 = tmp15 + tmp16
            tmp18 = libdevice.rsqrt(tmp17)
            tmp19 = tmp13 * tmp18
            tmp21 = tmp19 * tmp20
            tmp23 = tmp21 + tmp22
            tl.store(out_ptr2 + (r0_2 + 768 * x3), tmp23, r0_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_f32_cuda_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    convolution, patch_bias, weight, bias, shape_param = inputs
    convolution_t = _expect_f32_cuda_tensor(
        "convolution",
        convolution,
        CONV_SHAPE,
        CONV_STRIDE,
    )
    patch_bias_t = _expect_f32_cuda_tensor(
        "arg3_1",
        patch_bias,
        PATCH_BIAS_SHAPE,
        PATCH_BIAS_STRIDE,
    )
    weight_t = _expect_f32_cuda_tensor("arg4_1", weight, AFFINE_SHAPE, AFFINE_STRIDE)
    bias_t = _expect_f32_cuda_tensor("arg5_1", bias, AFFINE_SHAPE, AFFINE_STRIDE)

    if _shape_tuple(shape_param) != VIEW_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    tensors = (convolution_t, patch_bias_t, weight_t, bias_t)
    if any(tensor.device != convolution_t.device for tensor in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return convolution_t, patch_bias_t, weight_t, bias_t


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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_conv_patch_layernorm.py")

    convolution, patch_bias, weight, bias = _validate_inputs(inputs)
    with torch.cuda.device(convolution.device):
        output_3d = torch.empty_strided(
            OUTPUT_3D_SHAPE,
            OUTPUT_3D_STRIDE,
            device=convolution.device,
            dtype=torch.float32,
        )
        raw_stream = get_raw_stream(convolution.device.index or torch.cuda.current_device())
        _conv_patch_layernorm_kernel.run(
            convolution,
            patch_bias,
            weight,
            bias,
            output_3d,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
        output = torch.ops.aten._unsafe_view.default(output_3d, OUTPUT_FLAT_SHAPE)
    return output


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
