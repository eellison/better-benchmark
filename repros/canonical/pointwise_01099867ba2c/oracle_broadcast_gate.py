"""Gap diagnosis (classification: RECOMPUTE_FUSION): this oracle computes the full NFNet broadcast gate `add_109 + convolution_77 * sigmoid(convolution_79) * arg228_1 * 0.4` with a Triton schedule that evaluates the `[N,C,1,1]` sigmoid once per `(N,C)` row and reuses it for the full 6x6 spatial tile while writing the eager-compatible output layout directly; Inductor currently fuses the expression as a generic pointwise kernel over all output elements, so the broadcasted sigmoid producer is inlined and recomputed for each of the 36 spatial positions; Inductor cannot do this today because pointwise fusion lacks a broadcast-invariant tiling/reuse transform for expensive producers; the fix is RECOMPUTE_FUSION: teach the scheduler to tile pointwise broadcasts by invariant dimensions and materialize/reuse expensive scalar producers within each output tile."""
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
BATCH = 128
CHANNELS = 1536
HEIGHT = 6
WIDTH = 6
HW = HEIGHT * WIDTH
ROWS = BATCH * CHANNELS
INPUT_1X1_SHAPE = (BATCH, CHANNELS, 1, 1)
INPUT_6X6_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
CONTIGUOUS_6X6_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
CHANNELS_LAST_6X6_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
CONTIGUOUS_1X1_STRIDE = (CHANNELS, 1, 1, 1)
NUM_OUTPUTS = ROWS * HW
CLASSIFICATION = "RECOMPUTE_FUSION"

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 256}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1024}, num_warps=8, num_stages=3),
        ],
        key=["total_rows", "dtype_key"],
    )
    @triton.jit
    def _make_multiplier_kernel(
        gate_ptr,
        scale_ptr,
        multiplier_ptr,
        gate_stride_n: tl.constexpr,
        gate_stride_c: tl.constexpr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        dtype_key: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < total_rows

        n = rows // channels
        c = rows - n * channels

        gate_offsets = n * gate_stride_n + c * gate_stride_c
        gate = tl.load(gate_ptr + gate_offsets, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32) * 0.4
        multiplier = tl.sigmoid(gate) * scale
        tl.store(multiplier_ptr + rows, multiplier, mask=row_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        ],
        key=["layout_key", "dtype_key"],
    )
    @triton.jit
    def _apply_multiplier_kernel(
        x_ptr,
        add_ptr,
        multiplier_ptr,
        out_ptr,
        total_outputs: tl.constexpr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        layout_key: tl.constexpr,
        dtype_key: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_outputs
        if layout_key == 0:
            multiplier_offsets = offsets // hw
        else:
            n = offsets // (channels * hw)
            c = offsets - (offsets // channels) * channels
            multiplier_offsets = n * channels + c

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        multiplier = tl.load(multiplier_ptr + multiplier_offsets, mask=mask, other=0.0)
        out = residual + x * multiplier
        tl.store(out_ptr + offsets, out, mask=mask)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    allowed_dtypes: tuple[torch.dtype, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype not in allowed_dtypes:
        raise TypeError(f"{name} has dtype {value.dtype}, expected one of {allowed_dtypes}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    gate, x, scale, residual = inputs
    gate_t = _expect_tensor(
        "convolution_79",
        gate,
        INPUT_1X1_SHAPE,
        (torch.float32, torch.float16),
    )
    x_t = _expect_tensor(
        "convolution_77",
        x,
        INPUT_6X6_SHAPE,
        (torch.float32, torch.float16),
    )
    scale_t = _expect_tensor("arg228_1", scale, (), (torch.float32, torch.float16))
    residual_t = _expect_tensor(
        "add_109",
        residual,
        INPUT_6X6_SHAPE,
        (torch.float32, torch.float16),
    )

    tensors = (gate_t, x_t, scale_t, residual_t)
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise TypeError("all tensor inputs must have the same dtype for this repro scope")
    if any(tensor.device != gate_t.device for tensor in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if tuple(gate_t.stride()) != CONTIGUOUS_1X1_STRIDE:
        raise ValueError(
            f"convolution_79 has stride {tuple(gate_t.stride())}, "
            f"expected {CONTIGUOUS_1X1_STRIDE}"
        )
    allowed_6x6_strides = (CONTIGUOUS_6X6_STRIDE, CHANNELS_LAST_6X6_STRIDE)
    if tuple(x_t.stride()) not in allowed_6x6_strides:
        raise ValueError(
            f"convolution_77 has stride {tuple(x_t.stride())}, "
            f"expected one of {allowed_6x6_strides}"
        )
    if tuple(residual_t.stride()) != tuple(x_t.stride()):
        raise ValueError(
            f"add_109 stride {tuple(residual_t.stride())} does not match "
            f"convolution_77 stride {tuple(x_t.stride())}"
        )
    return gate_t, x_t, scale_t, residual_t


@oracle_impl(hardware="H100", shapes="(T([128, 1536, 1, 1], f32), T([128, 1536, 6, 6], f32), T([], f32), T([128, 1536, 6, 6], f32))")
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
        raise RuntimeError("Triton is required for oracle_broadcast_gate.py")

    gate, x, scale, residual = _validate_inputs(inputs)
    output = torch.empty_strided(
        INPUT_6X6_SHAPE,
        residual.stride(),
        device=residual.device,
        dtype=residual.dtype,
    )
    multiplier = torch.empty((ROWS,), device=residual.device, dtype=torch.float32)
    dtype_key = 0 if output.dtype == torch.float32 else 1
    make_grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_ROWS"]),)
    _make_multiplier_kernel[make_grid](
        gate,
        scale,
        multiplier,
        gate_stride_n=gate.stride(0),
        gate_stride_c=gate.stride(1),
        total_rows=ROWS,
        channels=CHANNELS,
        dtype_key=dtype_key,
    )
    layout_key = 0 if tuple(output.stride()) == CONTIGUOUS_6X6_STRIDE else 1
    apply_grid = lambda meta: (triton.cdiv(NUM_OUTPUTS, meta["BLOCK_SIZE"]),)
    _apply_multiplier_kernel[apply_grid](
        x,
        residual,
        multiplier,
        output,
        total_outputs=NUM_OUTPUTS,
        channels=CHANNELS,
        hw=HW,
        layout_key=layout_key,
        dtype_key=dtype_key,
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
