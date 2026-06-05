"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle implements the full GPT-J permute/contiguous-clone/view/view scope as a single storage-linear device copy from the strided f32[1,16,128,256] input into fresh contiguous [128,4096] output storage, whereas the best Inductor configuration is already within measurement noise of this storage-linear materialization on the measured shape; Inductor cannot get a confirmed local win here because the remaining work is a required dense copy plus output metadata, and the oracle does not establish a faster floor; the fix is BANDWIDTH_BOUND: record this layout-materialization case as at floor and revisit only if broader copy/layout codegen changes beat the current compiled kernel on the exact full scope."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


IN_SHAPE = (1, 16, 128, 256)
IN_STRIDE = (524288, 256, 4096, 1)
VIEW0_SHAPE = (1, 128, 4096)
OUT_SHAPE = (128, 4096)
OUT_STRIDE = (4096, 1)


def _shape_tuple(value):
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs):
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    getitem_220, shape_param_0, shape_param_1 = inputs
    if not isinstance(getitem_220, torch.Tensor):
        raise TypeError(f"getitem_220 must be a tensor, got {type(getitem_220)!r}")
    if not getitem_220.is_cuda:
        raise RuntimeError("CUDA input is required")
    if getitem_220.dtype != torch.float32:
        raise TypeError(f"expected getitem_220 dtype torch.float32, got {getitem_220.dtype}")
    if tuple(getitem_220.shape) != IN_SHAPE:
        raise ValueError(f"unexpected getitem_220 shape: {tuple(getitem_220.shape)}")
    if tuple(getitem_220.stride()) != IN_STRIDE:
        raise ValueError(f"unexpected getitem_220 stride: {tuple(getitem_220.stride())}")
    if getitem_220.storage_offset() != 0:
        raise ValueError(f"unexpected getitem_220 storage offset: {getitem_220.storage_offset()}")
    if _shape_tuple(shape_param_0) != VIEW0_SHAPE:
        raise ValueError(f"unexpected first view shape: {shape_param_0!r}")
    if _shape_tuple(shape_param_1) != OUT_SHAPE:
        raise ValueError(f"unexpected final view shape: {shape_param_1!r}")
    return getitem_220


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
    getitem_220 = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=getitem_220.device,
        dtype=torch.float32,
    )
    storage_order_view = torch.as_strided(getitem_220, OUT_SHAPE, OUT_STRIDE)
    output.copy_(storage_order_view)
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
