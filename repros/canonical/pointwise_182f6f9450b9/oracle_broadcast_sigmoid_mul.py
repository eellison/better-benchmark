"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete squeeze-excitation pointwise scope with a dense-layout Triton fast path that preserves the eager output stride and indexes the broadcast gate as `flat_offset // (H*W)`, whereas Inductor lowers the fused sigmoid/mul through a more generic expanded `[B,C,H,W]` pointwise schedule with heavier broadcast/layout indexing; Inductor cannot do this today because its pointwise scheduler/codegen does not specialize fused broadcast producers for the captured dense channel-gate layout; the fix is SCHEDULER_FUSION: teach fused pointwise scheduling to emit layout-specialized broadcast indexing for dense squeeze-excitation gates while retaining the generic strided fallback."""
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


CLASSIFICATION = "SCHEDULER_FUSION"


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
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        ],
        key=["N_ELEMENTS", "SPATIAL", "DTYPE_TAG"],
    )
    @triton.jit
    def _sigmoid_mul_flat_dense_kernel(
        gate_ptr,
        act_ptr,
        out_ptr,
        N_ELEMENTS: tl.constexpr,
        SPATIAL: tl.constexpr,
        DTYPE_TAG: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        gate = tl.load(gate_ptr + offsets // SPATIAL, mask=mask, other=0.0)
        act = tl.load(act_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, act * tl.sigmoid(gate), mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        ],
        key=["N_ELEMENTS", "CHANNELS", "SPATIAL", "DTYPE_TAG"],
    )
    @triton.jit
    def _sigmoid_mul_flat_contiguous_kernel(
        gate_ptr,
        act_ptr,
        out_ptr,
        N_ELEMENTS: tl.constexpr,
        CHANNELS: tl.constexpr,
        SPATIAL: tl.constexpr,
        GATE_STRIDE_B: tl.constexpr,
        GATE_STRIDE_C: tl.constexpr,
        DTYPE_TAG: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        bc = offsets // SPATIAL
        batch = bc // CHANNELS
        channel = bc - batch * CHANNELS
        gate = tl.load(
            gate_ptr + batch * GATE_STRIDE_B + channel * GATE_STRIDE_C,
            mask=mask,
            other=0.0,
        )
        act = tl.load(act_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, act * tl.sigmoid(gate), mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 1, "BLOCK_HW": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 2, "BLOCK_HW": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 4, "BLOCK_HW": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 256}, num_warps=8, num_stages=4),
        ],
        key=["BATCH", "CHANNELS", "SPATIAL", "DTYPE_TAG", "LAYOUT_TAG"],
    )
    @triton.jit
    def _sigmoid_mul_broadcast_kernel(
        gate_ptr,
        act_ptr,
        out_ptr,
        BATCH: tl.constexpr,
        CHANNELS: tl.constexpr,
        SPATIAL: tl.constexpr,
        GATE_STRIDE_B: tl.constexpr,
        GATE_STRIDE_C: tl.constexpr,
        ACT_STRIDE_B: tl.constexpr,
        ACT_STRIDE_C: tl.constexpr,
        ACT_STRIDE_H: tl.constexpr,
        ACT_STRIDE_W: tl.constexpr,
        OUT_STRIDE_B: tl.constexpr,
        OUT_STRIDE_C: tl.constexpr,
        OUT_STRIDE_H: tl.constexpr,
        OUT_STRIDE_W: tl.constexpr,
        WIDTH: tl.constexpr,
        DTYPE_TAG: tl.constexpr,
        LAYOUT_TAG: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_block = tl.program_id(1)
        hw_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        h_offsets = hw_offsets // WIDTH
        w_offsets = hw_offsets - h_offsets * WIDTH

        c_mask = c_offsets < CHANNELS
        hw_mask = hw_offsets < SPATIAL
        mask = c_mask[:, None] & hw_mask[None, :]

        gate = tl.load(
            gate_ptr + batch * GATE_STRIDE_B + c_offsets * GATE_STRIDE_C,
            mask=c_mask,
            other=0.0,
        )
        scale = tl.sigmoid(gate)

        act_offsets = (
            batch * ACT_STRIDE_B
            + c_offsets[:, None] * ACT_STRIDE_C
            + h_offsets[None, :] * ACT_STRIDE_H
            + w_offsets[None, :] * ACT_STRIDE_W
        )
        out_offsets = (
            batch * OUT_STRIDE_B
            + c_offsets[:, None] * OUT_STRIDE_C
            + h_offsets[None, :] * OUT_STRIDE_H
            + w_offsets[None, :] * OUT_STRIDE_W
        )

        act = tl.load(act_ptr + act_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + out_offsets, act * scale[:, None], mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_broadcast_sigmoid_mul.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    gate, activation = inputs
    if not isinstance(gate, torch.Tensor) or not isinstance(activation, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor inputs")
    if gate.device.type != "cuda" or activation.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if gate.dtype != activation.dtype:
        raise TypeError(f"expected matching dtypes, got {gate.dtype} and {activation.dtype}")
    if gate.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected float16 or float32 inputs, got {gate.dtype}")
    if gate.ndim != 4 or activation.ndim != 4:
        raise ValueError(f"expected rank-4 inputs, got {gate.ndim} and {activation.ndim}")
    if tuple(gate.shape[:2]) != tuple(activation.shape[:2]) or tuple(gate.shape[2:]) != (1, 1):
        raise ValueError(f"unexpected broadcast shapes: gate={tuple(gate.shape)} activation={tuple(activation.shape)}")
    if gate.stride(2) < 1 or gate.stride(3) < 1:
        raise ValueError(f"unexpected gate strides: {tuple(gate.stride())}")

    return gate, activation


def _layout_tag(tensor: torch.Tensor) -> int:
    if tensor.stride(1) == 1:
        return 1
    if tensor.stride(-1) == 1:
        return 0
    return 2


def oracle_forward(inputs):
    """Run the full broadcast sigmoid-multiply scope."""
    gate, activation = _validate_inputs(inputs)
    batch, channels, height, width = activation.shape
    output = torch.empty_strided(
        tuple(activation.shape),
        tuple(activation.stride()),
        device=activation.device,
        dtype=activation.dtype,
    )

    dtype_tag = 0 if activation.dtype is torch.float32 else 1
    if activation.is_contiguous() and output.is_contiguous():
        n_elements = activation.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        if gate.is_contiguous():
            _sigmoid_mul_flat_dense_kernel[grid](
                gate,
                activation,
                output,
                N_ELEMENTS=n_elements,
                SPATIAL=height * width,
                DTYPE_TAG=dtype_tag,
            )
        else:
            _sigmoid_mul_flat_contiguous_kernel[grid](
                gate,
                activation,
                output,
                N_ELEMENTS=n_elements,
                CHANNELS=channels,
                SPATIAL=height * width,
                GATE_STRIDE_B=gate.stride(0),
                GATE_STRIDE_C=gate.stride(1),
                DTYPE_TAG=dtype_tag,
            )
    else:
        grid = lambda meta: (
            batch,
            triton.cdiv(channels, meta["BLOCK_C"]),
            triton.cdiv(height * width, meta["BLOCK_HW"]),
        )
        _sigmoid_mul_broadcast_kernel[grid](
            gate,
            activation,
            output,
            BATCH=batch,
            CHANNELS=channels,
            SPATIAL=height * width,
            GATE_STRIDE_B=gate.stride(0),
            GATE_STRIDE_C=gate.stride(1),
            ACT_STRIDE_B=activation.stride(0),
            ACT_STRIDE_C=activation.stride(1),
            ACT_STRIDE_H=activation.stride(2),
            ACT_STRIDE_W=activation.stride(3),
            OUT_STRIDE_B=output.stride(0),
            OUT_STRIDE_C=output.stride(1),
            OUT_STRIDE_H=output.stride(2),
            OUT_STRIDE_W=output.stride(3),
            WIDTH=width,
            DTYPE_TAG=dtype_tag,
            LAYOUT_TAG=_layout_tag(activation),
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
