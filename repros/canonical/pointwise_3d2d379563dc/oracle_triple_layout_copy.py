"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Longformer permute-clone-view scope by materializing the three required fresh contiguous `[8192, 768]` outputs in one Triton layout-copy kernel that shares each `arg0_1[batch, seq, dim]` load across all three stores, whereas tuned Inductor already reaches the same practical layout-copy envelope for this full scope; Inductor cannot materially improve this repro through local fusion because the three non-aliasing clone outputs require the dominant output write traffic and the input read is small by comparison; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader layout-copy bandwidth or launch-overhead improvements move both implementations together."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 8
SEQ = 1024
DIM = 768
OUT_ROWS = BATCH * SEQ
OUT_SHAPE = (OUT_ROWS, DIM)
OUT_STRIDE = (DIM, 1)
IN_SHAPE = (BATCH, SEQ, DIM)
IN_STRIDE = (SEQ * DIM, DIM, 1)
BLOCK_ROWS = 8

if triton is not None:

    @triton.jit
    def _triple_layout_copy_kernel(
        x_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        BLOCK_M: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < 8192
        batch = rows % 8
        seq = rows // 8

        cols0 = tl.arange(0, 512)
        in0 = batch[:, None] * 786432 + seq[:, None] * 768 + cols0[None, :]
        out0 = rows[:, None] * 768 + cols0[None, :]
        mask0 = row_mask[:, None]
        vals0 = tl.load(x_ptr + in0, mask=mask0, other=0.0)
        tl.store(out0_ptr + out0, vals0, mask=mask0)
        tl.store(out1_ptr + out0, vals0, mask=mask0)
        tl.store(out2_ptr + out0, vals0, mask=mask0)

        cols1 = 512 + tl.arange(0, 256)
        in1 = batch[:, None] * 786432 + seq[:, None] * 768 + cols1[None, :]
        out1 = rows[:, None] * 768 + cols1[None, :]
        vals1 = tl.load(x_ptr + in1, mask=mask0, other=0.0)
        tl.store(out0_ptr + out1, vals1, mask=mask0)
        tl.store(out1_ptr + out1, vals1, mask=mask0)
        tl.store(out2_ptr + out1, vals1, mask=mask0)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected view shape as list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


@oracle_impl(hardware="H100", shapes="(T([8, 1024, 768], f32), S([8192, 768]), S([8192, 768]), S([8192, 768]))")
def oracle_forward(inputs):
    """Run the full permute-clone-view computation with three fresh outputs."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    x, shape0, shape1, shape2 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if tuple(x.shape) != IN_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != IN_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(x.stride())}")
    if x.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA input")

    out_shapes = (_shape_tuple(shape0), _shape_tuple(shape1), _shape_tuple(shape2))
    if out_shapes != (OUT_SHAPE, OUT_SHAPE, OUT_SHAPE):
        raise ValueError(f"unexpected view shapes: {out_shapes}")

    out0 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x.device, dtype=x.dtype)
    out1 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x.device, dtype=x.dtype)
    out2 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x.device, dtype=x.dtype)
    _triple_layout_copy_kernel[(triton.cdiv(OUT_ROWS, BLOCK_ROWS),)](
        x,
        out0,
        out1,
        out2,
        BLOCK_M=BLOCK_ROWS,
        num_warps=8,
    )
    return (out0, out1, out2)


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
