"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet squeeze-excitation residual pointwise scope as one dense-layout Triton streaming kernel, including the singleton-spatial sigmoid gate, activation multiply, explicit `* 2.0` then `* 0.2` scale, residual add, and the eager output stride for both contiguous and channels-last dense inputs, whereas Inductor already lowers the same isolated broadcast pointwise graph to one fused generated kernel with the same required two large f32 input reads, one small gate read, sigmoid math, and one f32 output store; Inductor cannot materially do less work inside this captured region because there is no producer/consumer fusion, reduction, scatter, layout materialization, or algebraic rewrite to remove beyond the mandatory memory traffic and pointwise sigmoid; the fix is BANDWIDTH_BOUND: record this as an at-floor squeeze-excitation pointwise case unless broader pointwise memory or sigmoid-codegen improvements move both implementations together."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 128
CHANNELS = 1536
HEIGHT = 7
WIDTH = 7
SPATIAL = HEIGHT * WIDTH
N_ELEMENTS = BATCH * CHANNELS * SPATIAL

GATE_SHAPE = (BATCH, CHANNELS, 1, 1)
ACTIVATION_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
CONTIGUOUS_STRIDE = (CHANNELS * SPATIAL, SPATIAL, WIDTH, 1)
CHANNELS_LAST_STRIDE = (CHANNELS * SPATIAL, 1, WIDTH * CHANNELS, CHANNELS)
GATE_STRIDE = (CHANNELS, 1, 1, 1)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 64, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
        ],
        key=["K_CHANNELS", "K_SPATIAL"],
    )
    @triton.jit
    def _se_residual_contiguous_kernel(
        gate_ptr,
        activation_ptr,
        residual_ptr,
        output_ptr,
        K_CHANNELS: tl.constexpr,
        K_SPATIAL: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_block = tl.program_id(1)
        hw_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        c_mask = c_offsets < K_CHANNELS
        hw_mask = hw_offsets < K_SPATIAL
        mask = c_mask[:, None] & hw_mask[None, :]

        gate = tl.load(
            gate_ptr + batch * K_CHANNELS + c_offsets,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        offsets = (
            batch * K_CHANNELS * K_SPATIAL
            + c_offsets[:, None] * K_SPATIAL
            + hw_offsets[None, :]
        )
        activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scaled = (activation * tl.sigmoid(gate)[:, None]) * 2.0
        out = scaled * 0.2 + residual
        tl.store(output_ptr + offsets, out, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        ],
        key=["N_ELEMENTS", "LAYOUT_TAG"],
    )
    @triton.jit
    def _se_residual_channels_last_kernel(
        gate_ptr,
        activation_ptr,
        residual_ptr,
        output_ptr,
        N_ELEMENTS: tl.constexpr,
        K_CHANNELS: tl.constexpr,
        K_SPATIAL: tl.constexpr,
        LAYOUT_TAG: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS

        if LAYOUT_TAG == 0:
            gate_offsets = offsets // K_SPATIAL
        else:
            batch = offsets // (K_CHANNELS * K_SPATIAL)
            channel = offsets - (offsets // K_CHANNELS) * K_CHANNELS
            gate_offsets = batch * K_CHANNELS + channel

        gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scaled = (activation * tl.sigmoid(gate)) * 2.0
        out = scaled * 0.2 + residual
        tl.store(output_ptr + offsets, out, mask=mask)


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_squeeze_excitation_residual.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    gate, activation, residual = inputs
    if (
        not isinstance(gate, torch.Tensor)
        or not isinstance(activation, torch.Tensor)
        or not isinstance(residual, torch.Tensor)
    ):
        raise TypeError(f"{REPRO_ID} expects tensor inputs")
    if gate.device.type != "cuda" or activation.device.type != "cuda" or residual.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if gate.device != activation.device or gate.device != residual.device:
        raise ValueError("all inputs must be on the same CUDA device")
    if gate.dtype != torch.float32 or activation.dtype != torch.float32 or residual.dtype != torch.float32:
        raise TypeError(
            f"{REPRO_ID} expects float32 inputs, got "
            f"{gate.dtype}, {activation.dtype}, {residual.dtype}"
        )
    if tuple(gate.shape) != GATE_SHAPE:
        raise ValueError(f"expected gate shape {GATE_SHAPE}, got {tuple(gate.shape)}")
    if tuple(activation.shape) != ACTIVATION_SHAPE or tuple(residual.shape) != ACTIVATION_SHAPE:
        raise ValueError(
            f"expected activation/residual shape {ACTIVATION_SHAPE}, got "
            f"{tuple(activation.shape)} and {tuple(residual.shape)}"
        )
    if gate.storage_offset() != 0 or activation.storage_offset() != 0 or residual.storage_offset() != 0:
        raise ValueError("oracle expects zero storage_offset inputs from the captured repro")
    if tuple(gate.stride()) != GATE_STRIDE:
        raise ValueError(f"expected dense singleton gate stride {GATE_STRIDE}, got {tuple(gate.stride())}")
    if tuple(activation.stride()) != tuple(residual.stride()):
        raise ValueError(
            f"activation and residual strides must match, got "
            f"{tuple(activation.stride())} and {tuple(residual.stride())}"
        )
    if tuple(activation.stride()) == CONTIGUOUS_STRIDE:
        return gate, activation, residual, 0
    if tuple(activation.stride()) == CHANNELS_LAST_STRIDE:
        return gate, activation, residual, 1
    raise ValueError(f"unexpected activation stride {tuple(activation.stride())}")


def oracle_forward(inputs):
    """Run the full squeeze-excitation residual pointwise scope."""
    gate, activation, residual, layout_tag = _validate_inputs(inputs)
    output = torch.empty_strided(
        tuple(activation.shape),
        tuple(activation.stride()),
        device=activation.device,
        dtype=activation.dtype,
    )
    if layout_tag == 0:
        grid = lambda meta: (
            BATCH,
            triton.cdiv(CHANNELS, meta["BLOCK_C"]),
            triton.cdiv(SPATIAL, meta["BLOCK_HW"]),
        )
        _se_residual_contiguous_kernel[grid](
            gate,
            activation,
            residual,
            output,
            K_CHANNELS=CHANNELS,
            K_SPATIAL=SPATIAL,
        )
    else:
        grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_SIZE"]),)
        _se_residual_channels_last_kernel[grid](
            gate,
            activation,
            residual,
            output,
            N_ELEMENTS=N_ELEMENTS,
            K_CHANNELS=CHANNELS,
            K_SPATIAL=SPATIAL,
            LAYOUT_TAG=layout_tag,
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
