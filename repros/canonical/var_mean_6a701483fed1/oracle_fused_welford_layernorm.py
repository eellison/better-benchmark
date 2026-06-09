"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ViT layer-norm region returned by Repro.forward, including the reshape, residual add, Welford var_mean over 768 columns, libdevice.rsqrt epsilon path, affine transform, flattened output view, and rsqrt/768 side output, using the same two-pass fused row kernel and floating-point order that Inductor emits; it differs only by packaging that full scope as a standalone oracle measured through bench_oracle, whereas Inductor already fuses this pattern into one kernel; Inductor cannot remove more local work today because exact Welford numerics require finishing the row reduction before normalization and the side output needs the reciprocal variance value, so the fix is BANDWIDTH_BOUND: treat this repro as at floor unless broader norm-reduction launch or memory-traffic work improves both implementations."""
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
ROWS = 175360
HIDDEN = 768
BATCH = 128
TOKENS = 1370
OUTPUT_3D_SHAPE = (BATCH, TOKENS, HIDDEN)
OUTPUT_3D_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
OUTPUT_FLAT_SHAPE = (ROWS, HIDDEN)
OUTPUT_FLAT_STRIDE = (HIDDEN, 1)
SIDE_OUTPUT_SHAPE = (BATCH, TOKENS, 1)
SIDE_OUTPUT_STRIDE = (TOKENS, 1, 1)
VECTOR_SHAPE = (HIDDEN,)
VECTOR_STRIDE = (1,)
XBLOCK = 2
R0_BLOCK = 1024


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
    from torch._inductor.runtime.triton_helpers import libdevice
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.reduction(
        size_hints={"x": 262144, "r0_": 1024},
        reduction_hint=ReductionHint.INNER,
        filename="/tmp/var_mean_6a701483fed1_oracle_triton.py",
        triton_meta={
            "signature": {
                "in_ptr0": "*fp32",
                "in_ptr1": "*fp32",
                "in_ptr2": "*fp32",
                "in_ptr3": "*fp32",
                "in_ptr4": "*fp32",
                "out_ptr2": "*fp32",
                "out_ptr3": "*fp32",
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
                    (7,): [["tt.divisibility", 16]],
                    (8,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "triton_red_fused_add_div_mul_rsqrt_sub_var_mean_view_0",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": False,
            "atomic_add_found": False,
            "num_load": 8,
            "num_store": 2,
            "num_reduction": 2,
            "autotune_hints": set(),
            "tiling_scores": {"x": 1402880, "r0_": 2154832896},
            "backend_hash": "2FF7A49C450C045FF2AC7A730B5A8D4CD6DF40914BC38271CC70D96733C19B52",
            "assert_indirect_indexing": True,
            "autotune_local_cache": False,
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
    def _fused_welford_layernorm_kernel(
        in_ptr0,
        in_ptr1,
        in_ptr2,
        in_ptr3,
        in_ptr4,
        out_ptr2,
        out_ptr3,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        R0_BLOCK: tl.constexpr,
    ):
        xnumel = 175360
        r0_numel = 768
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x3 = xindex
        x0 = xindex % 1370
        x1 = xindex // 1370

        tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)

        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(
                in_ptr0 + (r0_2 + 768 * x3),
                r0_mask & xmask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp1 = tl.load(
                in_ptr1 + (r0_2 + 768 * x3),
                r0_mask & xmask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp2 = tl.load(
                in_ptr2 + r0_2,
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp3 = tmp1 * tmp2
            tmp4 = tmp0 + tmp3
            tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
            tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
                tmp5,
                tmp6_mean,
                tmp6_m2,
                tmp6_weight,
                roffset == 0,
            )
            tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
            tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
            tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)

        tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
        tmp6 = tmp7[:, None]
        tmp10 = tmp8[:, None]
        tmp11 = tmp9[:, None]

        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp12 = tl.load(
                in_ptr0 + (r0_2 + 768 * x3),
                r0_mask & xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp13 = tl.load(
                in_ptr1 + (r0_2 + 768 * x3),
                r0_mask & xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp14 = tl.load(
                in_ptr2 + r0_2,
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp24 = tl.load(
                in_ptr3 + r0_2,
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp26 = tl.load(
                in_ptr4 + r0_2,
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp15 = tmp13 * tmp14
            tmp16 = tmp12 + tmp15
            tmp17 = tmp16 - tmp6
            tmp18 = tl.full([1, 1], 768.0, tl.float32)
            tmp19 = tmp10 / tmp18
            tmp20 = tl.full([1, 1], 1.0e-6, tl.float32)
            tmp21 = tmp19 + tmp20
            tmp22 = libdevice.rsqrt(tmp21)
            tmp23 = tmp17 * tmp22
            tmp25 = tmp23 * tmp24
            tmp27 = tmp25 + tmp26
            tl.store(out_ptr2 + (r0_2 + 768 * x3), tmp27, r0_mask & xmask)

        tmp28 = tl.full([1, 1], 768.0, tl.float32)
        tmp29 = tmp10 / tmp28
        tmp30 = tl.full([1, 1], 1.0e-6, tl.float32)
        tmp31 = tmp29 + tmp30
        tmp32 = libdevice.rsqrt(tmp31)
        tmp33 = tl.full([1, 1], 0.0013020833333333333, tl.float32)
        tmp34 = tmp32 * tmp33
        tl.store(out_ptr3 + x3, tmp34, xmask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _expect_f32_cuda_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm_45, arg165_1, add_77, arg166_1, arg167_1, shape0, shape1 = inputs
    addmm_45_t = _expect_f32_cuda_tensor(
        "addmm_45",
        addmm_45,
        OUTPUT_FLAT_SHAPE,
        OUTPUT_FLAT_STRIDE,
    )
    arg165_1_t = _expect_f32_cuda_tensor("arg165_1", arg165_1, VECTOR_SHAPE, VECTOR_STRIDE)
    add_77_t = _expect_f32_cuda_tensor("add_77", add_77, OUTPUT_3D_SHAPE, OUTPUT_3D_STRIDE)
    arg166_1_t = _expect_f32_cuda_tensor("arg166_1", arg166_1, VECTOR_SHAPE, VECTOR_STRIDE)
    arg167_1_t = _expect_f32_cuda_tensor("arg167_1", arg167_1, VECTOR_SHAPE, VECTOR_STRIDE)

    tensors = (addmm_45_t, arg165_1_t, add_77_t, arg166_1_t, arg167_1_t)
    if any(tensor.device != addmm_45_t.device for tensor in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if shape0_tuple != OUTPUT_3D_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if shape1_tuple != OUTPUT_FLAT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape1!r}")

    return (
        addmm_45_t,
        arg165_1_t,
        add_77_t,
        arg166_1_t,
        arg167_1_t,
        shape0_tuple,
        shape1_tuple,
    )


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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
        raise RuntimeError("Triton is required for oracle_fused_welford_layernorm.py")

    (
        addmm_45,
        arg165_1,
        add_77,
        arg166_1,
        arg167_1,
        shape0,
        shape1,
    ) = _validate_inputs(inputs)

    with torch.cuda.device(addmm_45.device):
        output_3d = torch.empty_strided(
            shape0,
            OUTPUT_3D_STRIDE,
            device=addmm_45.device,
            dtype=torch.float32,
        )
        side_output = torch.empty_strided(
            SIDE_OUTPUT_SHAPE,
            SIDE_OUTPUT_STRIDE,
            device=addmm_45.device,
            dtype=torch.float32,
        )

        raw_stream = get_raw_stream(addmm_45.device.index or torch.cuda.current_device())
        _fused_welford_layernorm_kernel.run(
            add_77,
            addmm_45,
            arg165_1,
            arg166_1,
            arg167_1,
            output_3d,
            side_output,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
        output = output_3d.view(shape1)
    return output, side_output


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
