"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete AlexNet training ReLU plus low-memory 3x3 stride-2 maxpool-with-offsets plus full ReLU<=0 mask scope by sinking ReLU into the stencil maxpool kernel and materializing the bool mask directly from the original input, whereas Inductor currently lowers the shared ReLU producer, maxpool value/index stencil, and sibling mask materialization through generic scheduling with avoidable ReLU materialization or producer replay boundaries; Inductor cannot do this today because its scheduler does not fuse one pointwise producer into both a stencil/indexing reduction consumer and a full-layout sibling materialization without splitting the region; the fix is SCHEDULER_FUSION: teach Inductor to keep ReLU virtual across low-memory maxpool-with-offsets and direct sibling mask stores for this layout/indexing pattern."""
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

BATCH = 1024
CHANNELS = 192
H_IN = 27
W_IN = 27
H_OUT = 13
W_OUT = 13
PLANES = BATCH * CHANNELS
INPUT_PLANE = H_IN * W_IN
OUTPUT_PLANE = H_OUT * W_OUT
TOTAL_INPUT = PLANES * INPUT_PLANE


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _relu_maxpool_kernel(
        input_ptr,
        values_ptr,
        offsets_ptr,
        BLOCK_POOL: tl.constexpr,
    ):
        plane = tl.program_id(0)
        input_base = plane * 729
        output_base = plane * 169

        pool_offsets = tl.arange(0, BLOCK_POOL)
        pool_mask = pool_offsets < 169
        oh = pool_offsets // 13
        ow = pool_offsets - oh * 13
        ih0 = oh * 2
        iw0 = ow * 2

        best_val = tl.full([BLOCK_POOL], -float("inf"), dtype=tl.float32)
        best_idx = tl.full([BLOCK_POOL], 0, dtype=tl.int8)

        for kh in tl.static_range(3):
            ih = ih0 + kh
            for kw in tl.static_range(3):
                iw = iw0 + kw
                raw = tl.load(
                    input_ptr + input_base + ih * 27 + iw,
                    mask=pool_mask,
                    other=-float("inf"),
                )
                relu = tl.where(raw != raw, raw, tl.maximum(raw, 0.0))
                is_better = pool_mask & ((relu > best_val) | (relu != relu))
                best_val = tl.where(is_better, relu, best_val)
                best_idx = tl.where(
                    is_better,
                    tl.full([BLOCK_POOL], kh * 3 + kw, dtype=tl.int8),
                    best_idx,
                )

        tl.store(values_ptr + output_base + pool_offsets, best_val, mask=pool_mask)
        tl.store(offsets_ptr + output_base + pool_offsets, best_idx, mask=pool_mask)

    @triton.jit
    def _relu_le_mask_kernel(
        input_ptr,
        mask_ptr,
        TOTAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < TOTAL
        raw = tl.load(input_ptr + offsets, mask=valid, other=1.0)
        tl.store(mask_ptr + offsets, raw <= 0.0, mask=valid)


@oracle_impl(hardware="H100", shapes="(T([1024, 192, 27, 27], f32))")
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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_relu_maxpool_mask.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (convolution_1,) = inputs
    if not isinstance(convolution_1, torch.Tensor):
        raise TypeError("oracle expects the input to be a tensor")
    if tuple(convolution_1.shape) != (BATCH, CHANNELS, H_IN, W_IN):
        raise ValueError(f"unexpected input shape: {tuple(convolution_1.shape)}")
    if convolution_1.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {convolution_1.dtype}")
    if convolution_1.device.type != "cuda":
        raise ValueError("oracle_relu_maxpool_mask.py expects a CUDA input")
    if not convolution_1.is_contiguous():
        raise ValueError(f"expected contiguous input, got strides {convolution_1.stride()}")

    values = torch.empty(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        device=convolution_1.device,
        dtype=torch.float32,
    )
    offsets = torch.empty(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        device=convolution_1.device,
        dtype=torch.int8,
    )
    le_mask = torch.empty(
        (BATCH, CHANNELS, H_IN, W_IN),
        device=convolution_1.device,
        dtype=torch.bool,
    )

    _relu_maxpool_kernel[(PLANES,)](
        convolution_1,
        values,
        offsets,
        BLOCK_POOL=256,
        num_warps=4,
    )
    _relu_le_mask_kernel[(triton.cdiv(TOTAL_INPUT, 1024),)](
        convolution_1,
        le_mask,
        TOTAL=TOTAL_INPUT,
        BLOCK=1024,
        num_warps=4,
    )
    return (values, offsets, le_mask)


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
