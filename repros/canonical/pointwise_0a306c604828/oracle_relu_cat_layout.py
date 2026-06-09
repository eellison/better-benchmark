"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete `relu(convolution_23)`, `relu(convolution_24)`, and channel-dimension `cat` scope with one paired Triton materialization kernel that maps each program to the same contiguous per-batch source offset in both inputs, applies ReLU, and writes both output channel halves directly in the final contiguous NCHW layout, whereas Inductor lowers the fused relu/cat as a generic one-output-element pointwise layout kernel with per-element concat-index decoding and source selection; Inductor cannot do this today because cat lowering has no paired equal-shape channel-concat template that writes multiple output segments from the same source coordinate; the fix is NEW_PATTERN: add a guarded concat-with-identical-pointwise-producers materialization template for equal-shape channel concatenations."""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "NEW_PATTERN"
BLOCK_SIZE = 1024

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


if triton is not None:

    @triton.jit
    def _relu_cat_paired_kernel(
        input0_ptr,
        input1_ptr,
        output_ptr,
        sample_elems: tl.constexpr,
        block_size: tl.constexpr,
    ):
        batch = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < sample_elems

        input_base = batch * sample_elems + offsets
        output_base = batch * (sample_elems * 2) + offsets

        values0 = tl.load(input0_ptr + input_base, mask=mask, other=0.0)
        values1 = tl.load(input1_ptr + input_base, mask=mask, other=0.0)
        relu0 = tl.where(values0 < 0.0, 0.0, values0)
        relu1 = tl.where(values1 < 0.0, 0.0, values1)

        tl.store(output_ptr + output_base, relu0, mask=mask)
        tl.store(output_ptr + output_base + sample_elems, relu1, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two tensor inputs, got {len(inputs)}")

    input0, input1 = inputs
    if not isinstance(input0, torch.Tensor) or not isinstance(input1, torch.Tensor):
        raise TypeError("oracle_relu_cat_layout.py expects two tensor inputs")
    if input0.shape != input1.shape:
        raise ValueError(f"input shapes must match, got {tuple(input0.shape)} and {tuple(input1.shape)}")
    if input0.ndim != 4:
        raise ValueError(f"expected NCHW rank-4 inputs, got rank {input0.ndim}")
    if input0.dtype != input1.dtype:
        raise TypeError(f"input dtypes must match, got {input0.dtype} and {input1.dtype}")
    if input0.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"unexpected input dtype {input0.dtype}")
    if input0.device != input1.device or input0.device.type != "cuda":
        raise ValueError("oracle_relu_cat_layout.py expects CUDA inputs on the same device")
    if not input0.is_contiguous() or not input1.is_contiguous():
        raise ValueError(
            "oracle_relu_cat_layout.py expects contiguous inputs from this repro's shape config"
        )

    return input0, input1


def _output_shape_and_stride(input0: torch.Tensor) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    batch, channels, height, width = input0.shape
    out_channels = channels * 2
    shape = (batch, out_channels, height, width)
    stride = (out_channels * height * width, height * width, width, 1)
    return shape, stride


@oracle_impl(hardware="H100", shapes="(T([512, 256, 13, 13], f16), T([512, 256, 13, 13], f16))")
def oracle_forward(inputs):
    """Run the full Repro.forward relu+channel-cat scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_relu_cat_layout.py")

    input0, input1 = _validate_inputs(inputs)
    output_shape, output_stride = _output_shape_and_stride(input0)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=input0.device,
        dtype=input0.dtype,
    )

    sample_elems = input0.shape[1] * input0.shape[2] * input0.shape[3]
    grid = (input0.shape[0], triton.cdiv(sample_elems, BLOCK_SIZE))
    _relu_cat_paired_kernel[grid](
        input0,
        input1,
        output,
        sample_elems=sample_elems,
        block_size=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return output


def _check_layout(output: torch.Tensor, input0: torch.Tensor) -> bool:
    output_shape, output_stride = _output_shape_and_stride(input0)
    return (
        tuple(output.shape) == output_shape
        and tuple(output.stride()) == output_stride
        and output.dtype == input0.dtype
        and output.storage_offset() == 0
    )


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
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out, inputs[0])
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()} "
            f"dtype={layout_out.dtype})"
        )
        ok = ok and layout_ok
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
