"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeBERTa query/key/value layout materialization and divide-by-eight epilogue as one direct Triton kernel from the captured `[4096, 1536]` input to the fresh contiguous `[192, 512, 64]` output, preserving the clone/view/permute output contract, whereas Inductor lowers the view/permute/clone/view/permute/div/permute chain through generic layout materialization and misses the direct final-output head-layout map; Inductor cannot do this today because it has no guarded template for this scaled attention-head materialization shape that writes the final contiguous contract directly from the packed projection; the fix is NEW_PATTERN: add a scaled head-layout materialization template for this affine index map and scalar epilogue."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None
    get_raw_stream = None
    triton_helpers = None
    triton_heuristics = None
    DeviceProperties = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

IN_SHAPE = (4096, 1536)
IN_STRIDE = (1536, 1)
OUT_SHAPE = (192, 512, 64)
OUT_STRIDE = (32768, 64, 1)
SHAPE_PARAM_0 = (8, 512, 1536)
SHAPE_PARAM_1 = (8, 512, 24, -1)
SHAPE_PARAM_2 = (-1, 512, 64)
NUMEL = 6291456


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    def _device_properties() -> DeviceProperties:
        props = torch.cuda.get_device_properties(0)
        return DeviceProperties(
            type="cuda",
            index=0,
            multi_processor_count=props.multi_processor_count,
            cc=props.major * 10 + props.minor,
            major=props.major,
            regs_per_multiprocessor=props.regs_per_multiprocessor,
            max_threads_per_multi_processor=props.max_threads_per_multi_processor,
            max_threads_per_block=props.max_threads_per_block,
            warp_size=props.warp_size,
        )

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={"x": 8388608},
        filename=__file__,
        triton_meta={
            "signature": {
                "addmm_1": "*fp32",
                "out": "*fp32",
                "xnumel": "i32",
                "XBLOCK": "constexpr",
            },
            "device": _device_properties(),
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
            "kernel_name": "oracle_layout_div_kernel",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": False,
            "atomic_add_found": False,
            "num_load": 1,
            "num_store": 1,
            "num_reduction": 0,
            "autotune_hints": set(),
            "tiling_scores": {"x": 75497472},
            "backend_hash": "oracle_pointwise_86e6de8ee102",
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
        min_elem_per_thread=0,
    )
    @triton.jit
    def _layout_div_kernel(
        addmm_1,
        out,
        xnumel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 6291456
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = xindex % 64
        x1 = (xindex // 64) % 512
        x2 = xindex // 32768
        x3 = xindex
        tmp0 = tl.load(addmm_1 + (x0 + 64 * (x2 % 24) + 1536 * x1 + 786432 * (x2 // 24)))
        tmp1 = tl.full([1], 0.125, tl.float32)
        tmp2 = tmp0 * tmp1
        tl.store(out + x3, tmp2)


def _check_shape_param(value: object, expected: tuple[int, ...], name: str) -> None:
    if tuple(value) != expected:
        raise ValueError(f"unexpected {name}: got={value!r} expected={expected!r}")


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    """Run the complete view/permute/clone/div/permute computation."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    addmm_1, shape_param_0, shape_param_1, shape_param_2 = inputs
    if not isinstance(addmm_1, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(addmm_1)!r}")
    if addmm_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if addmm_1.dtype != torch.float32:
        raise ValueError(f"unexpected input dtype: {addmm_1.dtype}")
    if tuple(addmm_1.shape) != IN_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(addmm_1.shape)}")
    if tuple(addmm_1.stride()) != IN_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(addmm_1.stride())}")
    _check_shape_param(shape_param_0, SHAPE_PARAM_0, "_shape_param_0")
    _check_shape_param(shape_param_1, SHAPE_PARAM_1, "_shape_param_1")
    _check_shape_param(shape_param_2, SHAPE_PARAM_2, "_shape_param_2")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=addmm_1.device,
        dtype=addmm_1.dtype,
    )
    _layout_div_kernel.run(
        addmm_1,
        out,
        NUMEL,
        stream=get_raw_stream(addmm_1.device.index or 0),
    )
    return out


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
