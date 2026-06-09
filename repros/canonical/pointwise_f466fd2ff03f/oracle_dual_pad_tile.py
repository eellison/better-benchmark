"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle hand-runs the full-scope fused output-space pad schedule for the shared `[16384, 50257]` input view, loading each valid element once and storing it into both the padded transpose and padded row-major outputs, whereas Inductor already emits the same one-kernel one-load/two-store materialization for this captured graph; Inductor cannot materially improve this isolated repro today because the returned tensors require the full input read, two full output writes, and the tiny zero tails with no removable intermediate left inside the scope; the fix is BANDWIDTH_BOUND: record this as an at-floor pad/layout case unless broader memory-codegen improvements or consumer fusion outside this repro removes a required materialized output."""
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
    from torch._inductor.runtime import triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties
except ImportError:
    triton = None
    tl = None
    get_raw_stream = None
    AsyncCompile = None
    triton_heuristics = None
    DeviceProperties = None

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


CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    _async_compile = AsyncCompile()
    _dual_pad_kernel = _async_compile.triton(
        "_dual_pad_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, TileHint
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"y": 16384, "x": 65536},
    tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "out_ptr0": "*fp32",
            "out_ptr1": "*fp32",
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
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid2D",
        "kernel_name": "_dual_pad_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 2,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"y": 6587678720, "x": 9881321472},
        "autotune_local_cache": True,
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
    },
    min_elem_per_thread=0,
)
@triton.jit
def _dual_pad_kernel(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 16384
    xnumel = 50260
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    valid = xindex < 50257
    values = tl.load(
        in_ptr0 + 50257 * yindex + xindex,
        mask=valid & xmask,
        eviction_policy="evict_last",
        other=0.0,
    )
    padded = tl.where(valid, values, tl.full([1, 1], 0.0, tl.float32))
    tl.store(out_ptr0 + yindex + 16384 * xindex, padded, mask=xmask)
    tl.store(out_ptr1 + xindex + 50260 * yindex, padded, mask=xmask)
""",
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile


def _expect_shape_param(shape_param, rows: int, cols: int) -> None:
    if list(shape_param) != [rows, cols]:
        raise ValueError(
            f"unexpected shape parameter {shape_param}, expected {[rows, cols]}"
        )


@oracle_impl(hardware="H100", shapes="(T([32, 512, 50257], f32), S([16384, 50257]))")
def oracle_forward(inputs):
    """Run the full view/permute/two-constant-pad repro scope."""
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")
    if triton is None or get_raw_stream is None:
        raise RuntimeError("Triton is required for this oracle")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor")
    if x.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects f32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA input")
    if x.dim() != 3:
        raise ValueError(f"{REPRO_ID} expects rank-3 input, got shape={tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layout")

    batch, seq, vocab = x.shape
    rows = batch * seq
    cols = vocab
    _expect_shape_param(shape_param, rows, cols)

    out_transposed = torch.empty_strided(
        (cols + 3, rows),
        (rows, 1),
        device=x.device,
        dtype=x.dtype,
    )
    out_copy = torch.empty_strided(
        (rows, cols + 3),
        (cols + 3, 1),
        device=x.device,
        dtype=x.dtype,
    )

    _dual_pad_kernel.run(
        x,
        out_transposed,
        out_copy,
        rows,
        cols + 3,
        stream=get_raw_stream(x.device.index),
    )
    return (out_transposed, out_copy)


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
