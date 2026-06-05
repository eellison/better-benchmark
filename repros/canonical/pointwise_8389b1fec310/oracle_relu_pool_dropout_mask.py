"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete AlexNet ReLU, low-memory 3x3 stride-2 maxpool offsets, identity adaptive_avg_pool/view, Inductor-style seed-index-0 dropout scaling, and full ReLU<=0 side mask by writing the pooled dropout tensor directly from a Triton stencil kernel plus a flat mask kernel, whereas Inductor already lowers the same scope into the practical stencil/dropout work and a simple sibling mask store with no large removable intermediate after the identity pool/view; Inductor cannot materially improve this today because the remaining cost is dominated by required overlapping maxpool reads, RNG/dropout stores, int8 offset stores, and the full input-sized bool mask rather than a missed scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, recompute-fusion, or new-pattern opportunity; the fix is BANDWIDTH_BOUND: record this as a bandwidth-floor diagnostic, with not_true_floor status when the stochastic dropout output is skipped by correctness checking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops

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


# --- Oracle kernel(s) ---
N_BATCH = 1024
CHANNELS = 256
H_IN = 13
W_IN = 13
H_OUT = 6
W_OUT = 6
INPUT_SHAPE = (N_BATCH, CHANNELS, H_IN, W_IN)
INPUT_STRIDE = (CHANNELS * H_IN * W_IN, H_IN * W_IN, W_IN, 1)
POOL_SHAPE = (N_BATCH, CHANNELS, H_OUT, W_OUT)
POOL_STRIDE = (CHANNELS * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1)
DROPOUT_SHAPE = (N_BATCH, CHANNELS * H_OUT * W_OUT)
DROPOUT_STRIDE = (CHANNELS * H_OUT * W_OUT, 1)
INPUT_NUMEL = N_BATCH * CHANNELS * H_IN * W_IN
POOL_NUMEL = N_BATCH * CHANNELS * H_OUT * W_OUT
SEED_COUNT = 2
DROPOUT_P = 0.5
DROPOUT_SCALE = 2.0
POOL_BLOCK = 256
MASK_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where(x <= 0.0, 0.0, x)


    @triton.jit
    def _pool_dropout_kernel(
        input_ptr,
        seed_ptr,
        offsets_ptr,
        dropout_ptr,
        total_pool: tl.constexpr,
        channels: tl.constexpr,
        h_in: tl.constexpr,
        w_in: tl.constexpr,
        h_out: tl.constexpr,
        w_out: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        out_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        active = out_offsets < total_pool

        ow = out_offsets % w_out
        tmp = out_offsets // w_out
        oh = tmp % h_out
        tmp = tmp // h_out
        channel = tmp % channels
        batch = tmp // channels

        input_base = (batch * channels + channel) * (h_in * w_in)
        best = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
        best_offset = tl.zeros((BLOCK_N,), dtype=tl.int32)

        for kh in tl.static_range(0, 3):
            ih = oh * 2 + kh
            for kw in tl.static_range(0, 3):
                iw = ow * 2 + kw
                raw = tl.load(
                    input_ptr + input_base + ih * w_in + iw,
                    mask=active,
                    other=-float("inf"),
                ).to(tl.float32)
                relu = _relu_preserve_nan(raw)
                take = active & ((relu > best) | (relu != relu))
                best = tl.where(take, relu, best)
                best_offset = tl.where(take, kh * 3 + kw, best_offset)

        seed = tl.load(seed_ptr)
        keep = tl.rand(seed, out_offsets.to(tl.uint32)) > dropout_p
        dropped = best * keep.to(tl.float32) * dropout_scale

        tl.store(offsets_ptr + out_offsets, best_offset.to(tl.int8), mask=active)
        tl.store(dropout_ptr + out_offsets, dropped, mask=active)


    @triton.jit
    def _relu_le_mask_kernel(
        input_ptr,
        le_ptr,
        total_input: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        active = offsets < total_input
        raw = tl.load(input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        tl.store(le_ptr + offsets, raw <= 0.0, mask=active)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence):
        raise TypeError(f"shape parameter must be a sequence, got {type(value).__name__}")
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    convolution_4, shape_param = inputs
    if not isinstance(convolution_4, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(convolution_4).__name__}")
    if not convolution_4.is_cuda:
        raise RuntimeError("CUDA input is required for this Triton oracle")
    if convolution_4.dtype is not torch.float32:
        raise TypeError(f"expected float32 input, got {convolution_4.dtype}")
    if tuple(convolution_4.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape {tuple(convolution_4.shape)}, expected {INPUT_SHAPE}")
    if tuple(convolution_4.stride()) != INPUT_STRIDE:
        raise ValueError(f"unexpected input stride {tuple(convolution_4.stride())}, expected {INPUT_STRIDE}")
    if _shape_tuple(shape_param) != DROPOUT_SHAPE:
        raise ValueError(f"unexpected view shape {shape_param!r}, expected {DROPOUT_SHAPE}")

    return convolution_4


def oracle_forward(inputs):
    """Run the full Repro.forward scope with direct maxpool/dropout and mask stores."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    convolution_4 = _validate_inputs(inputs)
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=convolution_4.device)
    pool_offsets = torch.empty_strided(
        POOL_SHAPE,
        POOL_STRIDE,
        device=convolution_4.device,
        dtype=torch.int8,
    )
    dropout = torch.empty_strided(
        DROPOUT_SHAPE,
        DROPOUT_STRIDE,
        device=convolution_4.device,
        dtype=torch.float32,
    )
    le_mask = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=convolution_4.device,
        dtype=torch.bool,
    )

    _pool_dropout_kernel[(triton.cdiv(POOL_NUMEL, POOL_BLOCK),)](
        convolution_4,
        seeds,
        pool_offsets,
        dropout,
        total_pool=POOL_NUMEL,
        channels=CHANNELS,
        h_in=H_IN,
        w_in=W_IN,
        h_out=H_OUT,
        w_out=W_OUT,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        BLOCK_N=POOL_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _relu_le_mask_kernel[(triton.cdiv(INPUT_NUMEL, MASK_BLOCK),)](
        convolution_4,
        le_mask,
        total_input=INPUT_NUMEL,
        BLOCK_N=MASK_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return pool_offsets, dropout, le_mask


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
