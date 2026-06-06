"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete `permute -> constant_pad_nd` scope by materializing the required fresh contiguous `[hidden, vocab + 3]` output with one pointwise transpose-pad kernel, whereas Inductor already lowers the same view-plus-pad graph to a single final-output materialization with only minor generic indexing overhead; Inductor cannot remove the dominant input read and output write traffic because the repro contract requires the transposed padded tensor, so the fix is BANDWIDTH_BOUND: record this case as at floor unless broader layout-copy memory codegen improves both implementations."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


PAD_COLS = 3
CLASSIFICATION = "BANDWIDTH_BOUND"
INPUT_SHAPE = (50265, 768)
OUTPUT_SHAPE = (768, 50268)
OUTPUT_STRIDE = (50268, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    _async_compile = AsyncCompile()
    _transpose_pad_kernel = _async_compile.triton(
        "_transpose_pad_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, TileHint
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"y": 1024, "x": 65536},
    tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "out_ptr0": "*fp32",
            "ynumel": "i32",
            "xnumel": "i32",
            "YBLOCK": "constexpr",
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
        "grid_type": "Grid2D",
        "kernel_name": "_transpose_pad_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"y": 154414080, "x": 308846592},
        "autotune_local_cache": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
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
def _transpose_pad_kernel(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 768
    xnumel = 50268
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    in_bounds = xindex < 50265
    values = tl.load(
        in_ptr0 + yindex + 768 * xindex,
        in_bounds & xmask & ymask,
        other=0.0,
    )
    values = tl.where(in_bounds, values, 0.0)
    tl.store(out_ptr0 + xindex + 50268 * yindex, values, xmask & ymask)
""",
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile


def _expect_input(inputs) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")
    value = inputs[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input must be a tensor, got {type(value)!r}")
    if value.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(value.shape)}")
    if tuple(value.shape) != INPUT_SHAPE:
        raise ValueError(f"{REPRO_ID} expects input shape {INPUT_SHAPE}, got {tuple(value.shape)}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects torch.float32 input, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if not value.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(value.stride())}")
    return value


def oracle_forward(inputs):
    """Run the exact full-scope transpose and right-pad materialization."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_transpose_pad.py")

    input_tensor = _expect_input(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    _transpose_pad_kernel.run(
        input_tensor,
        output,
        OUTPUT_SHAPE[0],
        OUTPUT_SHAPE[1],
        stream=get_raw_stream(input_tensor.device.index or 0),
    )
    return output


def _normalize_outputs(output) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in output:
            tensors.extend(_normalize_outputs(item))
        return tensors
    return []


def _check_layout_contract(instance, inputs) -> bool:
    with torch.no_grad():
        eager_outputs = _normalize_outputs(instance(*inputs))
        oracle_outputs = _normalize_outputs(oracle_forward(inputs))
    if any(output.is_cuda for output in oracle_outputs):
        torch.cuda.synchronize()

    input_storage_ptrs = {
        value.untyped_storage().data_ptr()
        for value in inputs
        if isinstance(value, torch.Tensor)
    }

    all_pass = True
    for index, (eager, oracle) in enumerate(zip(eager_outputs, oracle_outputs)):
        same_layout = (
            eager.dtype == oracle.dtype
            and tuple(eager.shape) == tuple(oracle.shape)
            and tuple(eager.stride()) == tuple(oracle.stride())
            and eager.storage_offset() == oracle.storage_offset()
        )
        eager_aliases_input = eager.untyped_storage().data_ptr() in input_storage_ptrs
        oracle_aliases_input = oracle.untyped_storage().data_ptr() in input_storage_ptrs
        same_aliasing = eager_aliases_input == oracle_aliases_input
        ok = same_layout and same_aliasing
        all_pass = all_pass and ok
        print(
            f"  output {index} layout: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(oracle.shape)} stride={tuple(oracle.stride())} "
            f"storage_offset={oracle.storage_offset()} aliases_input={oracle_aliases_input})"
        )
    return all_pass and len(eager_outputs) == len(oracle_outputs)


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
        ok = _check_layout_contract(instance, inputs) and ok
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
