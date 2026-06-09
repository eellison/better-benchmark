"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete `permute(0, 2, 1, 3).clone().view(32768, 192)` scope as one direct Triton layout-copy kernel from `arg270_1[n, h, s, d]` into the required fresh contiguous `[n * 64 + s, h * 48 + d]` output, vectorizing the small head dimension and writing the final view storage directly, whereas Inductor lowers the isolated materialization through a more generic layout-copy template that is measurably slower for this affine map; Inductor cannot do this today because layout-copy codegen does not specialize this N/H/S/D-to-N/S/(H*D) permutation into a compact head-vectorized copy; the fix is NEW_PATTERN: add a shape-aware layout-copy template for permute-clone-view patterns with small fixed head dimensions and direct final-view stores."""
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


N_BATCH = 512
N_HEADS = 4
N_SEQ = 64
N_DIM = 48
OUT_ROWS = N_BATCH * N_SEQ
OUT_COLS = N_HEADS * N_DIM
INPUT_SHAPE = (N_BATCH, N_HEADS, N_SEQ, N_DIM)
INPUT_STRIDE = (N_HEADS * N_SEQ * N_DIM, N_SEQ * N_DIM, N_DIM, 1)
OUTPUT_SHAPE = (OUT_ROWS, OUT_COLS)
OUTPUT_STRIDE = (OUT_COLS, 1)
BLOCK_ROWS = 16
BLOCK_DIM = 64

if triton is not None:

    @triton.jit
    def _layout_copy_kernel(
        input_ptr,
        output_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        dim = tl.arange(0, BLOCK_D)
        row_mask = rows < 32768
        dim_mask = dim < 48

        batch = rows // 64
        seq = rows - batch * 64
        input_base = batch[:, None] * 12288 + seq[:, None] * 48 + dim[None, :]
        output_base = rows[:, None] * 192 + dim[None, :]
        mask = row_mask[:, None] & dim_mask[None, :]

        values0 = tl.load(input_ptr + input_base, mask=mask, other=0.0)
        values1 = tl.load(input_ptr + input_base + 3072, mask=mask, other=0.0)
        values2 = tl.load(input_ptr + input_base + 6144, mask=mask, other=0.0)
        values3 = tl.load(input_ptr + input_base + 9216, mask=mask, other=0.0)

        tl.store(output_ptr + output_base, values0, mask=mask)
        tl.store(output_ptr + output_base + 48, values1, mask=mask)
        tl.store(output_ptr + output_base + 96, values2, mask=mask)
        tl.store(output_ptr + output_base + 144, values3, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


@oracle_impl(hardware="H100", shapes="(T([512, 4, 64, 48], f32), S([32768, 192]))")
def oracle_forward(inputs):
    """Run the complete permute-clone-view layout materialization."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != INPUT_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(x.stride())}")
    if x.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")
    if _shape_tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter: {shape_param!r}")

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=x.dtype,
    )
    _layout_copy_kernel[(triton.cdiv(OUT_ROWS, BLOCK_ROWS),)](
        x,
        output,
        BLOCK_M=BLOCK_ROWS,
        BLOCK_D=BLOCK_DIM,
        num_warps=8,
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
