"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileViT permute-clone-view layout materialization as one shape-specialized Triton copy from contiguous `[512, 4, 256, 36]` input storage to the required fresh contiguous `[131072, 144]` output, flattening the final `[batch * token, head * dim]` storage and using a direct affine input map, whereas Inductor lowers the same isolated region through a more generic layout-copy path that leaves measurable indexing/codegen overhead; Inductor cannot do this today because layout-copy codegen does not specialize this small-head permute-clone-view family into a storage-linear final-view copy; the fix is NEW_PATTERN: add a direct final-view layout-copy template for N/H/T/D-to-N*T/H*D materializations with small fixed H and D."""
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
N_BATCH = 512
N_HEADS = 4
N_TOKENS = 256
N_WIDTH = 36
OUT_ROWS = N_BATCH * N_TOKENS
OUT_COLS = N_HEADS * N_WIDTH
OUT_SHAPE = (OUT_ROWS, OUT_COLS)
OUT_STRIDE = (OUT_COLS, 1)
IN_SHAPE = (N_BATCH, N_HEADS, N_TOKENS, N_WIDTH)
IN_STRIDE = (
    N_HEADS * N_TOKENS * N_WIDTH,
    N_TOKENS * N_WIDTH,
    N_WIDTH,
    1,
)
TOTAL_ELEMENTS = OUT_ROWS * OUT_COLS
BLOCK_ELEMENTS = 1024

if triton is not None:

    @triton.jit
    def _layout_clone_kernel(
        x,
        out,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < 18874368

        col = offsets % 144
        row = offsets // 144
        batch = row // 256
        token = row - batch * 256
        head = col // 36
        dim = col - head * 36

        in_offsets = batch * 36864 + head * 9216 + token * 36 + dim
        values = tl.load(x + in_offsets, mask=mask, other=0.0)
        tl.store(out + offsets, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


@oracle_impl(hardware="H100", shapes="(T([512, 4, 256, 36], f32), S([131072, 144]))")
def oracle_forward(inputs):
    """Run the complete permute-clone-view layout materialization."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA tensor input")
    if x.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {x.dtype}")
    if tuple(x.shape) != IN_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != IN_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(x.stride())}")
    if _shape_tuple(shape_param) != OUT_SHAPE:
        raise ValueError(f"unexpected shape input: {shape_param!r}")

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(TOTAL_ELEMENTS, BLOCK_ELEMENTS),)
    _layout_clone_kernel[grid](
        x,
        out,
        BLOCK_N=BLOCK_ELEMENTS,
        num_warps=4,
        num_stages=1,
    )
    return out


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
