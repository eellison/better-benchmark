"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Albert softmax-backward row update returned by Repro.forward, including the metadata-only `[512,512,512] -> [8,64,512,512]` view, f32 product, last-dimension row sum over K=512, exact `fma.rn.f32` epilogue, and final contiguous f32 `[512,512,512]` view in one Triton row kernel, whereas bench_oracle measures tuned Inductor at the same CUDAGraph replay floor for this fixed row-reduction epilogue; Inductor cannot materially improve this local repro through scheduler fusion because the required two full input reads, one output write, row reduction, and fma epilogue dominate; the fix is BANDWIDTH_BOUND: record as at_floor unless broader row-reduction memory-traffic or launch-overhead work moves both paths."""
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


BATCH = 8
HEADS = 64
Q_LEN = 512
K_LEN = 512
ROWS = BATCH * HEADS * Q_LEN
OUT_SHAPE = (512, 512, 512)
OUT_STRIDE = (512 * 512, 512, 1)
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
VIEW_STRIDE = (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
BMM_SHAPE = OUT_SHAPE
BMM_STRIDE = OUT_STRIDE
ARG_SHAPE = VIEW_SHAPE
ARG_STRIDE = VIEW_STRIDE
ROWS_PER_BLOCK = 2


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fma_rn_f32(a, b, c):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[a, b, c],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def oracle_kernel(
        bmm_ptr,
        arg_ptr,
        out_ptr,
        ROWS_PER_BLOCK_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS_PER_BLOCK_ + tl.arange(0, ROWS_PER_BLOCK_)[:, None]
        cols = tl.arange(0, BLOCK_K)[None, :]
        offsets = rows * BLOCK_K + cols

        bmm = tl.load(bmm_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
        arg = tl.load(arg_ptr + offsets, eviction_policy="evict_first").to(tl.float32)

        product = _mul_rn_f32(bmm, arg)
        row_sum = tl.sum(product, axis=1)[:, None].to(tl.float32)
        out = _fma_rn_f32(-arg, row_sum, product)

        tl.store(out_ptr + offsets, out)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, value: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(value)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_tensor(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
    expected_stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tuple(value.shape)}")
    if tuple(value.stride()) != expected_stride:
        raise ValueError(f"{name} stride mismatch: expected {expected_stride}, got {tuple(value.stride())}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype mismatch: expected torch.float32, got {value.dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    bmm_45, arg20_1, shape0, shape1 = inputs
    bmm = _validate_tensor("bmm_45", bmm_45, BMM_SHAPE, BMM_STRIDE)
    arg = _validate_tensor("arg20_1", arg20_1, ARG_SHAPE, ARG_STRIDE)
    if arg.device != bmm.device:
        raise ValueError("bmm_45 and arg20_1 must be on the same CUDA device")
    _validate_shape_param("_shape_param_0", shape0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", shape1, OUT_SHAPE)
    return bmm, arg


def oracle_forward(inputs):
    """Run the full-scope Albert softmax-backward row reduction oracle."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_albert_softmax_backward.py")

    bmm, arg = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(ROWS, ROWS_PER_BLOCK),)
    oracle_kernel[grid](
        bmm,
        arg,
        output,
        ROWS_PER_BLOCK_=ROWS_PER_BLOCK,
        BLOCK_K=K_LEN,
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
