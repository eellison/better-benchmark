"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full ReLU, 2x2 stride-2 low-memory maxpool-with-offsets, and input-shaped ReLU <= 0 mask in one output-tiled Triton kernel, whereas Inductor currently emits separate work for the maxpool stencil and the full-size boolean mask from the shared ReLU producer; Inductor cannot do this today because the scheduler does not fuse a stencil reduction consumer and a layout-preserving side-output consumer of the same pointwise producer into one loop nest; the fix is SCHEDULER_FUSION: add multi-output producer fusion for low-memory maxpool-with-offsets plus full-layout pointwise mask stores."""
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


BATCH = 128
CHANNELS = 512
H_IN = 28
W_IN = 28
H_OUT = 14
W_OUT = 14
INPUT_PLANE = H_IN * W_IN
OUTPUT_PLANE = H_OUT * W_OUT
INPUT_STRIDE = (CHANNELS * INPUT_PLANE, INPUT_PLANE, W_IN, 1)
OUTPUT_STRIDE = (CHANNELS * OUTPUT_PLANE, OUTPUT_PLANE, W_OUT, 1)
INPUT_SHAPE = (BATCH, CHANNELS, H_IN, W_IN)
OUTPUT_SHAPE = (BATCH, CHANNELS, H_OUT, W_OUT)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _relu_maxpool_mask_kernel(
        input_ptr,
        values_ptr,
        offsets_ptr,
        mask_ptr,
        total_outputs: tl.constexpr,
        block: tl.constexpr,
    ):
        linear = tl.program_id(0) * block + tl.arange(0, block)
        active = linear < total_outputs

        ow = linear % 14
        tmp = linear // 14
        oh = tmp % 14
        plane = tmp // 14

        input_base = plane * 784 + oh * 56 + ow * 2
        x00 = tl.load(input_ptr + input_base, mask=active, other=-float("inf"))
        x01 = tl.load(input_ptr + input_base + 1, mask=active, other=-float("inf"))
        x10 = tl.load(input_ptr + input_base + 28, mask=active, other=-float("inf"))
        x11 = tl.load(input_ptr + input_base + 29, mask=active, other=-float("inf"))

        y00 = tl.where((x00 > 0.0) | (x00 != x00), x00, 0.0)
        y01 = tl.where((x01 > 0.0) | (x01 != x01), x01, 0.0)
        y10 = tl.where((x10 > 0.0) | (x10 != x10), x10, 0.0)
        y11 = tl.where((x11 > 0.0) | (x11 != x11), x11, 0.0)

        best = y00
        best_offset = tl.zeros((block,), dtype=tl.int32)

        take = (y01 > best) | ((y01 != y01) & (best == best))
        best = tl.where(take, y01, best)
        best_offset = tl.where(take, 1, best_offset)

        take = (y10 > best) | ((y10 != y10) & (best == best))
        best = tl.where(take, y10, best)
        best_offset = tl.where(take, 2, best_offset)

        take = (y11 > best) | ((y11 != y11) & (best == best))
        best = tl.where(take, y11, best)
        best_offset = tl.where(take, 3, best_offset)

        tl.store(values_ptr + linear, best, mask=active)
        tl.store(offsets_ptr + linear, best_offset.to(tl.int8), mask=active)

        tl.store(mask_ptr + input_base, x00 <= 0.0, mask=active)
        tl.store(mask_ptr + input_base + 1, x01 <= 0.0, mask=active)
        tl.store(mask_ptr + input_base + 28, x10 <= 0.0, mask=active)
        tl.store(mask_ptr + input_base + 29, x11 <= 0.0, mask=active)


def _torch_oracle(convolution_9):
    relu = torch.relu(convolution_9)
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu, [2, 2], [2, 2], [0, 0], [1, 1], False
    )
    return values, offsets, relu <= 0


@oracle_impl(hardware="H100", shapes="(T([128, 512, 28, 28], f32))")
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
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (convolution_9,) = inputs
    if not isinstance(convolution_9, torch.Tensor):
        raise TypeError("oracle expects the input to be a tensor")
    if tuple(convolution_9.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(convolution_9.shape)}")
    if tuple(convolution_9.stride()) != INPUT_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(convolution_9.stride())}")
    if convolution_9.dtype is not torch.float32:
        raise TypeError(f"expected float32 input, got {convolution_9.dtype}")
    if not convolution_9.is_cuda:
        return _torch_oracle(convolution_9)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    values = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=convolution_9.device,
        dtype=torch.float32,
    )
    offsets = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=convolution_9.device,
        dtype=torch.int8,
    )
    le_mask = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=convolution_9.device,
        dtype=torch.bool,
    )

    total_outputs = values.numel()
    block = 256
    _relu_maxpool_mask_kernel[(triton.cdiv(total_outputs, block),)](
        convolution_9,
        values,
        offsets,
        le_mask,
        total_outputs=total_outputs,
        block=block,
        num_warps=4,
        num_stages=3,
    )
    return values, offsets, le_mask


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
