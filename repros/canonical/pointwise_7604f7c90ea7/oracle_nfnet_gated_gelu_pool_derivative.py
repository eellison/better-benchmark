"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet sigmoid-gated residual, exact-erf GELU scale, fixed 2x2 stride-2 avg_pool2d output, and sibling full-resolution GELU-derivative output in one output-stencil Triton kernel that loads each 2x2 input tile once and writes both returned tensors, whereas Inductor currently lowers the shared broadcast pointwise producer and the pooling/derivative consumers as generic scheduled work with either a materialized full-resolution producer or duplicated producer math; Inductor cannot do this today because scheduler fusion does not represent a fixed-window pooling consumer and a sibling full-resolution pointwise output as one multi-output stencil schedule over the same producer DAG; the fix is SCHEDULER_FUSION: teach avg-pool stencil codegen to inline broadcast pointwise producers while emitting sibling pointwise stores from the same tile."""
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


CLASSIFICATION = "SCHEDULER_FUSION"
BATCH = 128
CHANNELS = 512
IN_HEIGHT = 24
IN_WIDTH = 24
OUT_HEIGHT = 12
OUT_WIDTH = 12
IN_HW = IN_HEIGHT * IN_WIDTH
OUT_HW = OUT_HEIGHT * OUT_WIDTH
ROWS = BATCH * CHANNELS
N_OUTPUTS = ROWS * OUT_HW

GATE_SHAPE = (BATCH, CHANNELS, 1, 1)
GATE_STRIDE = (CHANNELS, 1, 1, 1)
SPATIAL_SHAPE = (BATCH, CHANNELS, IN_HEIGHT, IN_WIDTH)
SPATIAL_STRIDE = (CHANNELS * IN_HW, IN_HW, IN_WIDTH, 1)
POOL_SHAPE = (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
POOL_STRIDE = (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1)
DERIV_SHAPE = SPATIAL_SHAPE
DERIV_STRIDE = SPATIAL_STRIDE

INV_SQRT2 = 0.7071067811865476
INV_SQRT_2PI = 0.3989422804014327
NFNET_GAMMA = 1.7015043497085571
NFNET_OUT_SCALE = 0.9622504486493761


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _nfnet_values(
        activation,
        residual,
        gate,
        scale,
    ):
        value = activation * gate
        value = value * 2.0
        value = value * scale
        value = value * 0.2
        value = value + residual

        erf_term = tl.math.erf(value * 0.7071067811865476) + 1.0
        gelu = (value * 0.5) * erf_term
        pooled_input = (gelu * 1.7015043497085571) * 0.9622504486493761

        exp_term = tl.exp((value * value) * -0.5)
        derivative = (erf_term * 0.5) + (value * (exp_term * 0.3989422804014327))
        return pooled_input, derivative


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
        ],
        key=["n_outputs"],
    )
    @triton.jit
    def _gated_gelu_pool_derivative_kernel(
        gate_ptr,
        activation_ptr,
        scale_ptr,
        residual_ptr,
        pool_out_ptr,
        derivative_out_ptr,
        n_outputs: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_outputs

        out_spatial = offsets % 144
        row = offsets // 144
        oh = out_spatial // 12
        ow = out_spatial - oh * 12
        h0 = oh * 2
        w0 = ow * 2

        gate = tl.sigmoid(tl.load(gate_ptr + row, mask=mask, other=0.0).to(tl.float32))
        scale = tl.load(scale_ptr).to(tl.float32)
        base = row * 576 + h0 * 24 + w0

        activation00 = tl.load(activation_ptr + base, mask=mask, other=0.0).to(tl.float32)
        residual00 = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
        pool00, deriv00 = _nfnet_values(activation00, residual00, gate, scale)

        activation01 = tl.load(activation_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        residual01 = tl.load(residual_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        pool01, deriv01 = _nfnet_values(activation01, residual01, gate, scale)

        activation10 = tl.load(activation_ptr + base + 24, mask=mask, other=0.0).to(tl.float32)
        residual10 = tl.load(residual_ptr + base + 24, mask=mask, other=0.0).to(tl.float32)
        pool10, deriv10 = _nfnet_values(activation10, residual10, gate, scale)

        activation11 = tl.load(
            activation_ptr + base + 25,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        residual11 = tl.load(
            residual_ptr + base + 25,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        pool11, deriv11 = _nfnet_values(activation11, residual11, gate, scale)

        pooled = (pool00 + pool01 + pool10 + pool11) * 0.25
        tl.store(pool_out_ptr + offsets, pooled, mask=mask)
        tl.store(derivative_out_ptr + base, deriv00, mask=mask)
        tl.store(derivative_out_ptr + base + 1, deriv01, mask=mask)
        tl.store(derivative_out_ptr + base + 24, deriv10, mask=mask)
        tl.store(derivative_out_ptr + base + 25, deriv11, mask=mask)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    gate = _require_f32_tensor("convolution_23", inputs[0], GATE_SHAPE, GATE_STRIDE)
    activation = _require_f32_tensor("convolution_21", inputs[1], SPATIAL_SHAPE, SPATIAL_STRIDE)
    scale = _require_f32_tensor("arg69_1", inputs[2], (), ())
    residual = _require_f32_tensor("add_26", inputs[3], SPATIAL_SHAPE, SPATIAL_STRIDE)
    if any(t.device != gate.device for t in (activation, scale, residual)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return gate, activation, scale, residual


@oracle_impl(hardware="H100", shapes="(T([128, 512, 1, 1], f32), T([128, 512, 24, 24], f32), T([], f32), T([128, 512, 24, 24], f32))")
def oracle_forward(inputs):
    """Run the full fused gate/GELU/avg-pool plus sibling derivative scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_nfnet_gated_gelu_pool_derivative.py")

    gate, activation, scale, residual = _validate_inputs(inputs)
    pool_output = torch.empty_strided(
        POOL_SHAPE,
        POOL_STRIDE,
        device=activation.device,
        dtype=torch.float32,
    )
    derivative_output = torch.empty_strided(
        DERIV_SHAPE,
        DERIV_STRIDE,
        device=activation.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(N_OUTPUTS, meta["BLOCK_SIZE"]),)
    _gated_gelu_pool_derivative_kernel[grid](
        gate,
        activation,
        scale,
        residual,
        pool_output,
        derivative_output,
        n_outputs=N_OUTPUTS,
    )
    return (pool_output, derivative_output)


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
