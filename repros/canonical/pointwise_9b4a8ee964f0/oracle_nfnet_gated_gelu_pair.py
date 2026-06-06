"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet gated exact-GELU value and derivative pair in one layout-specialized Triton pointwise kernel, loading the `[128,1536,1,1]` sigmoid gate once per channel tile and broadcasting it across the 12x12 spatial tile while emitting both contiguous f32 outputs, whereas Inductor's CUDAGraph-measured compiled pointwise kernel reaches the same single-launch envelope for the decomposed sigmoid/mul/exact-erf-GELU/exp derivative graph; Inductor cannot materially do less local work for this captured scope because the remaining cost is the required f32 input/output traffic plus exact `erf`, `exp`, and sigmoid math rather than an avoidable reduction, scatter, materialization, or alias-preserving layout operation; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope pointwise case unless broader pointwise special-function codegen or launch-overhead work moves both implementations."""
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


BATCH = 128
CHANNELS = 1536
HEIGHT = 12
WIDTH = 12
SPATIAL = HEIGHT * WIDTH
OUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
GATE_SHAPE = (BATCH, CHANNELS, 1, 1)
OUT_STRIDE = (CHANNELS * SPATIAL, SPATIAL, WIDTH, 1)
GATE_STRIDE = (CHANNELS, 1, 1, 1)
N_CHANNEL_ROWS = BATCH * CHANNELS
INV_SQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327
NFNET_GAMMA = 1.7015043497085571
NFNET_BETA = 0.9622504486493761


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 1, "BLOCK_HW": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 2, "BLOCK_HW": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 4, "BLOCK_HW": 256}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 256}, num_warps=8, num_stages=4),
        ],
        key=["N_ROWS"],
    )
    @triton.jit
    def _gated_gelu_pair_kernel(
        gate_ptr,
        activation_ptr,
        scalar_ptr,
        residual_ptr,
        out_value_ptr,
        out_grad_ptr,
        N_ROWS: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)
        mask = (row_offsets[:, None] < N_ROWS) & (hw_offsets[None, :] < 144)

        gate = tl.load(gate_ptr + row_offsets, mask=row_offsets < N_ROWS, other=0.0).to(tl.float32)
        activation = tl.load(
            activation_ptr + row_offsets[:, None] * 144 + hw_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + row_offsets[:, None] * 144 + hw_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scalar = tl.load(scalar_ptr).to(tl.float32)

        mul_tensor = activation * tl.sigmoid(gate)[:, None]
        mul_tensor_1 = mul_tensor * 2.0
        mul_tensor_2 = mul_tensor_1 * scalar
        mul_tensor_3 = mul_tensor_2 * 0.2
        add_tensor = mul_tensor_3 + residual

        mul_tensor_4 = add_tensor * 0.5
        mul_tensor_5 = add_tensor * 0.7071067811865476
        erf_default = tl.math.erf(mul_tensor_5)
        add_tensor_1 = erf_default + 1.0
        mul_tensor_6 = mul_tensor_4 * add_tensor_1
        mul_tensor_7 = mul_tensor_6 * 1.7015043497085571
        mul_tensor_8 = mul_tensor_7 * 0.9622504486493761

        mul_tensor_9 = add_tensor_1 * 0.5
        mul_tensor_10 = add_tensor * add_tensor
        mul_tensor_11 = mul_tensor_10 * -0.5
        exp_default = tl.exp(mul_tensor_11)
        mul_tensor_12 = exp_default * 0.3989422804014327
        mul_tensor_13 = add_tensor * mul_tensor_12
        add_tensor_2 = mul_tensor_9 + mul_tensor_13

        out_offsets = row_offsets[:, None] * 144 + hw_offsets[None, :]
        tl.store(out_value_ptr + out_offsets, mul_tensor_8, mask=mask)
        tl.store(out_grad_ptr + out_offsets, add_tensor_2, mask=mask)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} has storage_offset {value.storage_offset()}, expected 0")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_nfnet_gated_gelu_pair.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    gate, activation, scalar, residual = inputs
    gate = _expect_tensor("convolution_36", gate, GATE_SHAPE, GATE_STRIDE)
    activation = _expect_tensor("convolution_34", activation, OUT_SHAPE, OUT_STRIDE)
    scalar = _expect_tensor("arg106_1", scalar, (), ())
    residual = _expect_tensor("add_45", residual, OUT_SHAPE, OUT_STRIDE)
    if gate.device != activation.device or gate.device != scalar.device or gate.device != residual.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return gate, activation, scalar, residual


def oracle_forward(inputs):
    """Run the full Repro.forward gated exact-GELU value/derivative scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    gate, activation, scalar, residual = _validate_inputs(inputs)
    out_value = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=activation.device,
        dtype=torch.float32,
    )
    out_grad = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=activation.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(N_CHANNEL_ROWS, meta["BLOCK_C"]),)
    _gated_gelu_pair_kernel[grid](
        gate,
        activation,
        scalar,
        residual,
        out_value,
        out_grad,
        N_ROWS=N_CHANNEL_ROWS,
    )
    return (out_value, out_grad)


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
