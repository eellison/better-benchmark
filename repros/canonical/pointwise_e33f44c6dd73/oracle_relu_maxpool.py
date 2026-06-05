"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full captured ReLU and 3x3 stride-2 no-padding low-memory maxpool-with-offsets tuple with a shape-specialized straight-line Triton lowering, whereas Inductor currently emits a generic fused pointwise-style maxpool lowering with modulo address decode and generic argmax tie handling; Inductor cannot do this today because codegen has no dedicated fixed-window no-padding low-memory maxpool-with-offsets pattern for this ReLU producer case; the fix is NEW_PATTERN: add a specialized lowering for static 3x3 stride-2 no-padding maxpool-with-offsets that emits direct address arithmetic and first-index argmax semantics."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


KERNEL_SIZE = 3
STRIDE = 2
PADDING = 0
BLOCK_N = 1024


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _relu(x):
        return tl.where(x != x, x, tl.maximum(x, 0.0))


    @triton.jit
    def _take_candidate(candidate, candidate_offset, best, best_offset):
        better = (candidate > best) | ((best == best) & (candidate != candidate))
        return tl.where(better, candidate, best), tl.where(better, candidate_offset, best_offset)


    @triton.jit
    def _relu_maxpool_kernel(
        input_ptr,
        values_ptr,
        offsets_ptr,
        HEIGHT: tl.constexpr,
        WIDTH: tl.constexpr,
        OUT_HEIGHT: tl.constexpr,
        OUT_WIDTH: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        """Fuse ReLU into a 3x3 stride-2 low-memory maxpool-with-offsets."""
        out_plane = OUT_HEIGHT * OUT_WIDTH
        input_plane = HEIGHT * WIDTH
        output_index = tl.program_id(0) * BLOCK_N_ + tl.arange(0, BLOCK_N_)

        out_w = output_index % OUT_WIDTH
        out_h = (output_index // OUT_WIDTH) % OUT_HEIGHT
        plane = output_index // out_plane
        input_index = plane * input_plane + out_h * (2 * WIDTH) + out_w * 2

        v0 = _relu(tl.load(input_ptr + input_index).to(tl.float32))
        v1 = _relu(tl.load(input_ptr + input_index + 1).to(tl.float32))
        v2 = _relu(tl.load(input_ptr + input_index + 2).to(tl.float32))
        v3 = _relu(tl.load(input_ptr + input_index + WIDTH).to(tl.float32))
        v4 = _relu(tl.load(input_ptr + input_index + WIDTH + 1).to(tl.float32))
        v5 = _relu(tl.load(input_ptr + input_index + WIDTH + 2).to(tl.float32))
        v6 = _relu(tl.load(input_ptr + input_index + 2 * WIDTH).to(tl.float32))
        v7 = _relu(tl.load(input_ptr + input_index + 2 * WIDTH + 1).to(tl.float32))
        v8 = _relu(tl.load(input_ptr + input_index + 2 * WIDTH + 2).to(tl.float32))

        best = v0
        best_offset = tl.full((BLOCK_N_,), 0, tl.int32)
        best, best_offset = _take_candidate(v1, 1, best, best_offset)
        best, best_offset = _take_candidate(v2, 2, best, best_offset)
        best, best_offset = _take_candidate(v3, 3, best, best_offset)
        best, best_offset = _take_candidate(v4, 4, best, best_offset)
        best, best_offset = _take_candidate(v5, 5, best, best_offset)
        best, best_offset = _take_candidate(v6, 6, best, best_offset)
        best, best_offset = _take_candidate(v7, 7, best, best_offset)
        best, best_offset = _take_candidate(v8, 8, best, best_offset)

        tl.store(values_ptr + output_index, best)
        tl.store(offsets_ptr + output_index, best_offset.to(tl.int8))


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_relu_maxpool.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"input must be a tensor, got {type(x)!r}")
    if not x.is_cuda:
        raise ValueError("oracle requires a CUDA input")
    if x.ndim != 4:
        raise ValueError(f"input must be rank 4 NCHW, got shape {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected float16 or float32 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous NCHW input, got strides {x.stride()}")
    if x.shape[2] < KERNEL_SIZE or x.shape[3] < KERNEL_SIZE:
        raise ValueError(f"spatial shape is too small for 3x3 maxpool: {tuple(x.shape)}")
    return x


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full ReLU and low-memory maxpool-with-offsets scope."""
    x = _validate_inputs(inputs)
    batch, channels, height, width = x.shape
    out_height = (height + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    out_width = (width + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    out_shape = (batch, channels, out_height, out_width)
    out_stride = (channels * out_height * out_width, out_height * out_width, out_width, 1)

    values = torch.empty_strided(out_shape, out_stride, device=x.device, dtype=x.dtype)
    offsets = torch.empty_strided(out_shape, out_stride, device=x.device, dtype=torch.int8)

    total_planes = batch * channels
    total_outputs = total_planes * out_height * out_width
    if total_outputs % BLOCK_N != 0:
        raise ValueError(f"output numel {total_outputs} is not divisible by BLOCK_N={BLOCK_N}")
    grid = (triton.cdiv(total_outputs, BLOCK_N),)
    _relu_maxpool_kernel[grid](
        x,
        values,
        offsets,
        HEIGHT=height,
        WIDTH=width,
        OUT_HEIGHT=out_height,
        OUT_WIDTH=out_width,
        BLOCK_N_=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    return values, offsets


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
