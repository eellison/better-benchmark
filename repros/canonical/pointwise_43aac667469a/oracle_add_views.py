"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full metadata-view plus elementwise-add scope as one storage-linear Triton pointwise kernel over the contiguous inputs/output, ignoring the logical 3D view shape after validating that the views are metadata-only, whereas Inductor's compiled repro keeps enough generic view/shape handling in the generated pointwise path to measure slower for the same two-read/one-write contract; Inductor cannot do this today because pointwise codegen does not consistently canonicalize metadata-only view pairs into a mask-free storage-linear add when the public output shape is rank-expanded; the fix is NEW_PATTERN: add a view-linearization pointwise canonicalization that emits direct contiguous indexing for metadata-only views before scheduling/codegen."""
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


# --- Oracle kernel(s) ---

if triton is not None:

    _async_compile = AsyncCompile()
    _add_views_kernel = _async_compile.triton(
        "_add_views_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"x": 8388608},
    filename=__file__,
    triton_meta={
        "signature": {
            "left_ptr": "*fp32",
            "right_ptr": "*fp32",
            "out_ptr": "*fp32",
            "xnumel": "i32",
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
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_add_views_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 2,
        "num_store": 1,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"x": 83886080},
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
def _add_views_kernel(left_ptr, right_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    left = tl.load(left_ptr + xindex, None)
    right = tl.load(right_ptr + xindex, None)
    tl.store(out_ptr + xindex, left + right, None)
""",
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile


def _shape_tuple(name: str, value) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc
    if len(shape) != 3:
        raise ValueError(f"{name} must be a 3D output shape, got {shape}")
    if any(dim < 0 for dim in shape):
        raise ValueError(f"{name} contains a negative dimension: {shape}")
    return shape


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _require_input_tensor(name: str, value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if value.dim() != 2:
        raise ValueError(f"{name} must be the captured 2D input layout, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {tuple(value.stride())}")
    return value


@oracle_impl(hardware="H100", shapes="(T([2048, 2560], f32), T([2048, 2560], f32), S([16, 128, 2560]), S([16, 128, 2560]))")
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
    if triton is None or get_raw_stream is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    left = _require_input_tensor("mm_6", inputs[0])
    right = _require_input_tensor("mm_8", inputs[1])
    out_shape = _shape_tuple("_shape_param_0", inputs[2])
    out_shape_1 = _shape_tuple("_shape_param_1", inputs[3])

    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"input shapes must match, got {tuple(left.shape)} and {tuple(right.shape)}")
    if tuple(left.stride()) != tuple(right.stride()):
        raise ValueError(f"input strides must match, got {tuple(left.stride())} and {tuple(right.stride())}")
    if out_shape != out_shape_1:
        raise ValueError(f"shape params must match, got {out_shape} and {out_shape_1}")

    n_elements = left.numel()
    expected_numel = 1
    for dim in out_shape:
        expected_numel *= dim
    if expected_numel != n_elements:
        raise ValueError(f"shape {out_shape} has {expected_numel} elements but inputs have {n_elements}")
    if n_elements % 4096 != 0:
        raise ValueError(f"{REPRO_ID} expects captured shapes with element count divisible by 4096")

    output = torch.empty_strided(
        out_shape,
        _contiguous_strides(out_shape),
        device=left.device,
        dtype=torch.float32,
    )
    device_index = left.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _add_views_kernel.run(
        left,
        right,
        output,
        n_elements,
        stream=get_raw_stream(device_index),
    )
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
