"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete SqueezeNet two-input ReLU, channel concat, and sibling `relu <= 0` mask scope in one paired Triton materialization kernel that loads each source element once and writes the final contiguous cat plus both bool outputs, whereas Inductor already lowers this fixed pointwise/cat region into the same mandatory dense read/write traffic envelope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a narrower new pattern because the required two f32 input reads, cat stores, and mask stores dominate the CUDAGraph timing; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise bandwidth or launch-overhead work moves both implementations."""
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
CLASSIFICATION = "BANDWIDTH_BOUND"
BATCH = 512
CHANNELS = 256
HEIGHT = 13
WIDTH = 13
SAMPLE_ELEMS = CHANNELS * HEIGHT * WIDTH
CAT_CHANNELS = CHANNELS * 2
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (SAMPLE_ELEMS, HEIGHT * WIDTH, WIDTH, 1)
CAT_SHAPE = (BATCH, CAT_CHANNELS, HEIGHT, WIDTH)
CAT_STRIDE = (CAT_CHANNELS * HEIGHT * WIDTH, HEIGHT * WIDTH, WIDTH, 1)
BLOCK_SIZE = 1024

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
    def _relu_cat_masks_kernel(
        input0_ptr,
        input1_ptr,
        cat_out_ptr,
        mask1_out_ptr,
        mask0_out_ptr,
        sample_elems: tl.constexpr,
        block_size: tl.constexpr,
    ):
        batch = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < sample_elems

        input_base = batch * sample_elems + offsets
        cat_base = batch * (sample_elems * 2) + offsets

        values0 = tl.load(input0_ptr + input_base, mask=mask, other=0.0)
        values1 = tl.load(input1_ptr + input_base, mask=mask, other=0.0)

        relu0 = tl.where(values0 < 0.0, 0.0, values0)
        relu1 = tl.where(values1 < 0.0, 0.0, values1)
        mask0 = values0 <= 0.0
        mask1 = values1 <= 0.0

        tl.store(cat_out_ptr + cat_base, relu0, mask=mask)
        tl.store(cat_out_ptr + cat_base + sample_elems, relu1, mask=mask)
        tl.store(mask1_out_ptr + input_base, mask1, mask=mask)
        tl.store(mask0_out_ptr + input_base, mask0, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two tensor inputs, got {len(inputs)}")
    input0, input1 = inputs
    if not isinstance(input0, torch.Tensor) or not isinstance(input1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects two tensor inputs")
    if tuple(input0.shape) != INPUT_SHAPE or tuple(input1.shape) != INPUT_SHAPE:
        raise ValueError(
            f"{REPRO_ID} expects input shapes {INPUT_SHAPE}, got "
            f"{tuple(input0.shape)} and {tuple(input1.shape)}"
        )
    if input0.dtype != torch.float32 or input1.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects float32 inputs")
    if input0.device != input1.device or input0.device.type != "cuda":
        raise ValueError(f"{REPRO_ID} expects both inputs on the same CUDA device")
    if tuple(input0.stride()) != INPUT_STRIDE or tuple(input1.stride()) != INPUT_STRIDE:
        raise ValueError(
            f"{REPRO_ID} expects contiguous NCHW input strides {INPUT_STRIDE}, got "
            f"{tuple(input0.stride())} and {tuple(input1.stride())}"
        )
    return input0, input1


def oracle_forward(inputs):
    """Run the full Repro.forward ReLU, channel-cat, and mask scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_relu_cat_masks.py")

    input0, input1 = _validate_inputs(inputs)
    cat_out = torch.empty_strided(
        CAT_SHAPE,
        CAT_STRIDE,
        device=input0.device,
        dtype=torch.float32,
    )
    mask1_out = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=input0.device,
        dtype=torch.bool,
    )
    mask0_out = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=input0.device,
        dtype=torch.bool,
    )

    grid = (BATCH, triton.cdiv(SAMPLE_ELEMS, BLOCK_SIZE))
    _relu_cat_masks_kernel[grid](
        input0,
        input1,
        cat_out,
        mask1_out,
        mask0_out,
        sample_elems=SAMPLE_ELEMS,
        block_size=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return (cat_out, mask1_out, mask0_out)


def _check_output_contract(outputs: tuple[torch.Tensor, ...]) -> bool:
    if len(outputs) != 3:
        return False
    cat_out, mask1_out, mask0_out = outputs
    expected = (
        (cat_out, CAT_SHAPE, CAT_STRIDE, torch.float32),
        (mask1_out, INPUT_SHAPE, INPUT_STRIDE, torch.bool),
        (mask0_out, INPUT_SHAPE, INPUT_STRIDE, torch.bool),
    )
    layout_ok = all(
        tuple(tensor.shape) == shape
        and tuple(tensor.stride()) == stride
        and tensor.dtype == dtype
        and tensor.storage_offset() == 0
        for tensor, shape, stride, dtype in expected
    )
    alias_ok = len({tensor.untyped_storage().data_ptr() for tensor in outputs}) == len(outputs)
    return layout_ok and alias_ok


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
            contract_outputs = oracle_forward(inputs)
            torch.cuda.synchronize()
        contract_ok = _check_output_contract(contract_outputs)
        print(
            f"  output contract: {'PASS' if contract_ok else 'FAIL'} "
            f"(cat_stride={contract_outputs[0].stride()} "
            f"mask_stride={contract_outputs[1].stride()})"
        )
        ok = ok and contract_ok
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
