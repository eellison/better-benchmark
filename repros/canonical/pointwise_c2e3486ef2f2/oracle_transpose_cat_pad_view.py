"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle preserves the exact eager MobileBERT `permute -> cat -> {constant_pad_nd, permute}` contract by materializing both the padded `[512,30524]` output and a separate contiguous unpadded cat backing buffer whose returned view has stride `(1, 30522)`, whereas Inductor's compiled schedule writes only the padded buffer and returns the second output as a cheaper alias view with stride `(1, 30524)`; Inductor cannot produce the eager stride from the padded storage because each padded row has two extra columns, so the exact contract requires the extra full cat-buffer store and is bandwidth-bound rather than a missing local fusion; the fix is BANDWIDTH_BOUND: no performance optimization is indicated for the exact contract, and an Inductor change would be a layout-correctness change that preserves the separate cat buffer if exact output stride/view metadata is required."""
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


CLASSIFICATION = "BANDWIDTH_BOUND"
ARG2_SHAPE = (30522, 128)
ARG1117_SHAPE = (384, 30522)
CAT_SHAPE = (512, 30522)
CAT_STRIDE = (30522, 1)
PAD_SHAPE = (512, 30524)
PAD_STRIDE = (30524, 1)
CAT_COLS = 30522
ARG2_ROWS_AFTER_PERMUTE = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    _async_compile = AsyncCompile()
    _cat_pad_view_kernel = _async_compile.triton(
        "_cat_pad_view_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, TileHint
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"y": 512, "x": 32768},
    tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "in_ptr1": "*fp32",
            "out_pad": "*fp32",
            "out_cat": "*fp32",
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
                (3,): [["tt.divisibility", 16]],
                (4,): [["tt.divisibility", 16]],
                (5,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid2D",
        "kernel_name": "_cat_pad_view_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 2,
        "num_store": 2,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"y": 15627264, "x": 171908096},
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
def _cat_pad_view_kernel(in_ptr0, in_ptr1, out_pad, out_cat, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 30524
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    in_cat = xindex < 30522
    from_arg2 = yindex < 128
    arg2_values = tl.load(
        in_ptr0 + 128 * xindex + yindex,
        in_cat & from_arg2 & xmask & ymask,
        other=0.0,
    )
    arg1117_values = tl.load(
        in_ptr1 + 30522 * (yindex - 128) + xindex,
        in_cat & ~from_arg2 & xmask & ymask,
        other=0.0,
    )
    values = tl.where(from_arg2, arg2_values, arg1117_values)
    values = tl.where(in_cat, values, 0.0)
    tl.store(out_pad + xindex + 30524 * yindex, values, xmask & ymask)
    tl.store(out_cat + xindex + 30522 * yindex, values, in_cat & xmask & ymask)
""",
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile


def _expect_tensor(value: object, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} dtype {value.dtype} does not match torch.float32")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def oracle_forward(inputs):
    """Run the oracle computation with the same inputs and outputs as Repro.forward."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_transpose_cat_pad_view.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg2_1 = _expect_tensor(inputs[0], ARG2_SHAPE, "arg2_1")
    arg1117_1 = _expect_tensor(inputs[1], ARG1117_SHAPE, "arg1117_1")
    padded = torch.empty_strided(
        PAD_SHAPE,
        PAD_STRIDE,
        device=arg2_1.device,
        dtype=torch.float32,
    )
    cat_buf = torch.empty_strided(
        CAT_SHAPE,
        CAT_STRIDE,
        device=arg2_1.device,
        dtype=torch.float32,
    )
    _cat_pad_view_kernel.run(
        arg2_1,
        arg1117_1,
        padded,
        cat_buf,
        PAD_SHAPE[0],
        PAD_SHAPE[1],
        stream=get_raw_stream(arg2_1.device.index),
    )
    return padded, cat_buf.permute(1, 0)


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
