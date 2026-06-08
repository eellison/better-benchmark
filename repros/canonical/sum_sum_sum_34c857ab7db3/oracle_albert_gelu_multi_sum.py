"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full ALBERT tanh-GELU backward fragment returned by `Repro.forward`, including the materialized GELU-backward tensor, its returned non-contiguous `[16384, 4096]` transpose view, the eleven sibling `[4096, 16384] -> [16384]` column reductions, and the final sibling add with the GELU column sum, using the same two-kernel pointwise-then-multi-reduction envelope and natural `libdevice.tanh` formulation that Inductor selects; it does not materially differ from Inductor because a single-pass write-plus-reduce attempt serializes too much row work and recomputing the tanh-GELU in the reduction would replace one required output read with two input reads plus expensive tanh math; Inductor cannot do less local work while it must return the full materialized GELU tensor and also reduce it; the fix is BANDWIDTH_BOUND: record this repro as at_floor unless broader pointwise, reduction, tanh, or memory-traffic improvements move both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None
    libdevice = None

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


ROWS = 4096
COLS = 16384
VIEW_SHAPE = (8, 512, COLS)
FLAT_SHAPE = (ROWS, COLS)
SUM_SHAPE = (COLS,)
TRANSPOSE_SHAPE = (COLS, ROWS)
TRANSPOSE_STRIDE = (1, COLS)

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

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={"x": 67108864},
        filename="/tmp/sum_sum_sum_34c857ab7db3_oracle_pointwise.py",
        triton_meta={
            "signature": {
                "in_ptr0": "*fp32",
                "in_ptr1": "*fp32",
                "out_ptr0": "*fp32",
                "xnumel": "i32",
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
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_albert_gelu_backward_pointwise",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": False,
            "atomic_add_found": False,
            "num_load": 2,
            "num_store": 1,
            "num_reduction": 0,
            "autotune_hints": set(),
            "tiling_scores": {"x": 1073741824},
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
        min_elem_per_thread=0,
    )
    @triton.jit
    def _inductor_style_gelu_pointwise(
        in_ptr0,
        in_ptr1,
        out_ptr0,
        xnumel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 67108864
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + x0, None)
        tmp1 = tl.load(in_ptr1 + x0, None)
        tmp2 = tl.full([1], 0.5, tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tmp1 * tmp1
        tmp6 = tmp5 * tmp1
        tmp7 = tl.full([1], 0.044715, tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp1 + tmp8
        tmp10 = tl.full([1], 0.7978845608028654, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = libdevice.tanh(tmp11)
        tmp13 = tmp12 * tmp12
        tmp14 = tl.full([1], 1.0, tl.float32)
        tmp15 = tmp14 - tmp13
        tmp16 = tmp4 * tmp15
        tmp17 = tmp16 * tmp10
        tmp18 = tmp17 * tmp7
        tmp19 = tl.full([1], 3.0, tl.float32)
        tmp20 = tmp5 * tmp19
        tmp21 = tmp18 * tmp20
        tmp22 = tmp17 + tmp21
        tmp23 = tmp12 + tmp14
        tmp24 = tmp0 * tmp23
        tmp25 = tmp24 * tmp2
        tmp26 = tmp22 + tmp25
        tl.store(out_ptr0 + x0, tmp26, None)

    @triton_heuristics.reduction(
        size_hints={"x": 16384, "r0_": 4096},
        reduction_hint=ReductionHint.DEFAULT,
        filename="/tmp/sum_sum_sum_34c857ab7db3_oracle_reduction.py",
        triton_meta={
            "signature": {
                "in_out_ptr0": "*fp32",
                "in_ptr0": "*fp32",
                "in_ptr1": "*fp32",
                "in_ptr2": "*fp32",
                "in_ptr3": "*fp32",
                "in_ptr4": "*fp32",
                "in_ptr5": "*fp32",
                "in_ptr6": "*fp32",
                "in_ptr7": "*fp32",
                "in_ptr8": "*fp32",
                "in_ptr9": "*fp32",
                "in_ptr10": "*fp32",
                "in_ptr11": "*fp32",
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
                    (9,): [["tt.divisibility", 16]],
                    (10,): [["tt.divisibility", 16]],
                    (11,): [["tt.divisibility", 16]],
                    (12,): [["tt.divisibility", 16]],
                    (13,): [["tt.divisibility", 16]],
                    (14,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_albert_gelu_multi_reduction",
            "mutated_arg_names": ["in_out_ptr0"],
            "optimize_mem": True,
            "no_x_dim": False,
            "atomic_add_found": False,
            "num_load": 12,
            "num_store": 1,
            "num_reduction": 12,
            "autotune_hints": set(),
            "tiling_scores": {"x": 3221356544, "r0_": 0},
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
    def _inductor_style_multi_reduction(
        in_out_ptr0,
        in_ptr0,
        in_ptr1,
        in_ptr2,
        in_ptr3,
        in_ptr4,
        in_ptr5,
        in_ptr6,
        in_ptr7,
        in_ptr8,
        in_ptr9,
        in_ptr10,
        in_ptr11,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        R0_BLOCK: tl.constexpr,
    ):
        xnumel = 16384
        r0_numel = 4096
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        x0 = xindex

        _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            tmp0 = tl.load(
                in_ptr0 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            _tmp2 = tl.where(r0_mask, _tmp2 + tmp0, _tmp2)
        tmp2 = tl.sum(_tmp2, 1)[:, None]

        _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp30 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp34 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        _tmp46 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            tmp4 = tl.load(
                in_ptr1 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp8 = tl.load(
                in_ptr2 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp12 = tl.load(
                in_ptr3 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp16 = tl.load(
                in_ptr4 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp20 = tl.load(
                in_ptr5 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp24 = tl.load(
                in_ptr6 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp28 = tl.load(
                in_ptr7 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp32 = tl.load(
                in_ptr8 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp36 = tl.load(
                in_ptr9 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp40 = tl.load(
                in_ptr10 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp44 = tl.load(
                in_ptr11 + (x0 + 16384 * r0_index),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            _tmp6 = tl.where(r0_mask, _tmp6 + tmp4, _tmp6)
            _tmp10 = tl.where(r0_mask, _tmp10 + tmp8, _tmp10)
            _tmp14 = tl.where(r0_mask, _tmp14 + tmp12, _tmp14)
            _tmp18 = tl.where(r0_mask, _tmp18 + tmp16, _tmp18)
            _tmp22 = tl.where(r0_mask, _tmp22 + tmp20, _tmp22)
            _tmp26 = tl.where(r0_mask, _tmp26 + tmp24, _tmp26)
            _tmp30 = tl.where(r0_mask, _tmp30 + tmp28, _tmp30)
            _tmp34 = tl.where(r0_mask, _tmp34 + tmp32, _tmp34)
            _tmp38 = tl.where(r0_mask, _tmp38 + tmp36, _tmp38)
            _tmp42 = tl.where(r0_mask, _tmp42 + tmp40, _tmp42)
            _tmp46 = tl.where(r0_mask, _tmp46 + tmp44, _tmp46)
        tmp6 = tl.sum(_tmp6, 1)[:, None]
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tmp14 = tl.sum(_tmp14, 1)[:, None]
        tmp18 = tl.sum(_tmp18, 1)[:, None]
        tmp22 = tl.sum(_tmp22, 1)[:, None]
        tmp26 = tl.sum(_tmp26, 1)[:, None]
        tmp30 = tl.sum(_tmp30, 1)[:, None]
        tmp34 = tl.sum(_tmp34, 1)[:, None]
        tmp38 = tl.sum(_tmp38, 1)[:, None]
        tmp42 = tl.sum(_tmp42, 1)[:, None]
        tmp46 = tl.sum(_tmp46, 1)[:, None]
        tmp48 = tmp2 + tmp6
        tmp49 = tmp48 + tmp10
        tmp50 = tmp49 + tmp14
        tmp51 = tmp50 + tmp18
        tmp52 = tmp51 + tmp22
        tmp53 = tmp52 + tmp26
        tmp54 = tmp53 + tmp30
        tmp55 = tmp54 + tmp34
        tmp56 = tmp55 + tmp38
        tmp57 = tmp56 + tmp42
        tmp58 = tmp57 + tmp46
        tl.store(in_out_ptr0 + x0, tmp58, None)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _expect_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != FLAT_SHAPE:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {FLAT_SHAPE}")
    if tuple(value.stride()) != (COLS, 1):
        raise ValueError(f"{name} stride {tuple(value.stride())} does not match {(COLS, 1)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype {value.dtype} does not match torch.float32")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 28:
        raise ValueError(f"{REPRO_ID} expects 28 inputs, got {len(inputs)}")

    tensors = tuple(_expect_tensor(f"input {i}", inputs[i]) for i in range(13))
    for i in range(13, 24):
        shape = _shape_tuple(inputs[i])
        if shape != SUM_SHAPE:
            raise ValueError(f"shape input {i} is {shape}, expected {SUM_SHAPE}")
    if _shape_tuple(inputs[24]) != VIEW_SHAPE:
        raise ValueError(f"shape input 24 is {inputs[24]!r}, expected {VIEW_SHAPE}")
    if _shape_tuple(inputs[25]) != VIEW_SHAPE:
        raise ValueError(f"shape input 25 is {inputs[25]!r}, expected {VIEW_SHAPE}")
    if _shape_tuple(inputs[26]) != FLAT_SHAPE:
        raise ValueError(f"shape input 26 is {inputs[26]!r}, expected {FLAT_SHAPE}")
    if _shape_tuple(inputs[27]) != SUM_SHAPE:
        raise ValueError(f"shape input 27 is {inputs[27]!r}, expected {SUM_SHAPE}")
    return tensors


def oracle_forward(inputs):
    """Run the full Repro.forward computation for the canonical ALBERT shape."""
    tensors = _validate_inputs(inputs)
    (
        view_15,
        view_45,
        view_75,
        view_105,
        view_135,
        view_165,
        view_195,
        view_225,
        view_255,
        view_285,
        view_315,
        mm_136,
        arg24_1,
    ) = tensors

    out_base = torch.empty(FLAT_SHAPE, device=mm_136.device, dtype=torch.float32)
    out_sum = torch.empty(SUM_SHAPE, device=mm_136.device, dtype=torch.float32)

    raw_stream = get_raw_stream(mm_136.device.index or 0)
    _inductor_style_gelu_pointwise.run(
        mm_136,
        arg24_1,
        out_base,
        ROWS * COLS,
        stream=raw_stream,
    )
    _inductor_style_multi_reduction.run(
        out_sum,
        view_15,
        view_45,
        view_75,
        view_105,
        view_135,
        view_165,
        view_195,
        view_225,
        view_255,
        view_285,
        view_315,
        out_base,
        COLS,
        ROWS,
        stream=raw_stream,
    )

    return (out_base.permute(1, 0), out_sum)


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
