"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ReLU followed by 2x2 stride-2 low-memory maxpool-with-offsets in one output-tiled Triton stencil kernel that writes the pooled values and int8 local offsets directly, whereas CUDAGraph-measured Inductor already emits the same fused ReLU/maxpool-with-offsets loop for this repro; Inductor cannot materially improve this local case with additional layout/stencil fusion because the mandatory four input reads plus value/offset stores dominate and no intermediate ReLU tensor is materialized; the fix is BANDWIDTH_BOUND: record this row as at floor unless a broader maxpool argmax codegen or memory-bandwidth improvement moves both implementations together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


KERNEL_SIZE = 2
STRIDE = 2
BLOCK_N = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _take_candidate(candidate, candidate_offset, best, best_offset):
        take = (candidate > best) | ((candidate != candidate) & (best == best))
        best = tl.where(take, candidate, best)
        best_offset = tl.where(take, candidate_offset, best_offset)
        return best, best_offset


    @triton.jit
    def _relu_maxpool2d_offsets_kernel(
        input_ptr,
        values_ptr,
        offsets_ptr,
        total_outputs: tl.constexpr,
        h_in: tl.constexpr,
        w_in: tl.constexpr,
        h_out: tl.constexpr,
        w_out: tl.constexpr,
        block: tl.constexpr,
    ):
        linear = tl.program_id(0) * block + tl.arange(0, block)
        active = linear < total_outputs

        out_w = linear % w_out
        tmp = linear // w_out
        out_h = tmp % h_out
        plane = tmp // h_out

        input_base = plane * h_in * w_in + out_h * 2 * w_in + out_w * 2
        x00 = tl.load(input_ptr + input_base, mask=active, other=-float("inf")).to(tl.float32)
        x01 = tl.load(input_ptr + input_base + 1, mask=active, other=-float("inf")).to(tl.float32)
        x10 = tl.load(input_ptr + input_base + w_in, mask=active, other=-float("inf")).to(tl.float32)
        x11 = tl.load(input_ptr + input_base + w_in + 1, mask=active, other=-float("inf")).to(tl.float32)

        best = _relu_preserve_nan(x00)
        best_offset = tl.zeros((block,), dtype=tl.int32)

        candidate = _relu_preserve_nan(x01)
        best, best_offset = _take_candidate(candidate, 1, best, best_offset)
        candidate = _relu_preserve_nan(x10)
        best, best_offset = _take_candidate(candidate, 2, best, best_offset)
        candidate = _relu_preserve_nan(x11)
        best, best_offset = _take_candidate(candidate, 3, best, best_offset)

        tl.store(values_ptr + linear, best, mask=active)
        tl.store(offsets_ptr + linear, best_offset.to(tl.int8), mask=active)


def _torch_oracle(convolution_9):
    relu = torch.ops.aten.relu.default(convolution_9)
    return torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu, [2, 2], [2, 2], [0, 0], [1, 1], False
    )


def _validate_input(inputs):
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (convolution_9,) = inputs
    if not isinstance(convolution_9, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(convolution_9)!r}")
    if convolution_9.dim() != 4:
        raise ValueError(f"expected a 4D NCHW input, got shape {tuple(convolution_9.shape)}")
    if not convolution_9.is_contiguous():
        raise ValueError(f"expected contiguous NCHW input, got stride {tuple(convolution_9.stride())}")
    if convolution_9.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected float16 or float32 input, got {convolution_9.dtype}")

    batch, channels, h_in, w_in = convolution_9.shape
    if h_in < KERNEL_SIZE or w_in < KERNEL_SIZE:
        raise ValueError(f"input spatial size is too small for 2x2 maxpool: {(h_in, w_in)}")

    h_out = (h_in - KERNEL_SIZE) // STRIDE + 1
    w_out = (w_in - KERNEL_SIZE) // STRIDE + 1
    return convolution_9, (batch, channels, h_in, w_in, h_out, w_out)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same output structure, shapes, dtypes, and strides as Repro()(*inputs).
    """
    convolution_9, shape = _validate_input(inputs)
    batch, channels, h_in, w_in, h_out, w_out = shape

    if not convolution_9.is_cuda:
        return _torch_oracle(convolution_9)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    output_shape = (batch, channels, h_out, w_out)
    output_stride = (channels * h_out * w_out, h_out * w_out, w_out, 1)
    values = torch.empty_strided(
        output_shape,
        output_stride,
        device=convolution_9.device,
        dtype=convolution_9.dtype,
    )
    offsets = torch.empty_strided(
        output_shape,
        output_stride,
        device=convolution_9.device,
        dtype=torch.int8,
    )

    total_outputs = values.numel()
    _relu_maxpool2d_offsets_kernel[(triton.cdiv(total_outputs, BLOCK_N),)](
        convolution_9,
        values,
        offsets,
        total_outputs=total_outputs,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
        block=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return values, offsets


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
