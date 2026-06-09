"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet gate/residual SiLU pointwise scope as one layout-preserving Triton materialization, using a dense flat fast path for the captured contiguous `Repro()(*make_inputs())` shape and a strided fallback for sibling layouts, whereas tuned Inductor already lowers the same expression to an equivalent one-launch fused pointwise kernel within CUDAGraph timing noise; Inductor cannot materially improve this isolated repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new pattern because the remaining work is dominated by mandatory f32 activation/residual reads, transcendental pointwise math, and the f32 output store; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise math/materialization case unless broader pointwise codegen or launch-overhead changes move both implementations."""
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


CLASSIFICATION = "BANDWIDTH_BOUND"
FINAL_SCALE = 0.9622504486493761


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
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=4),
        ],
        key=["N_ELEMENTS", "SPATIAL"],
    )
    @triton.jit
    def _nfnet_silu_gate_flat_kernel(
        gate_ptr,
        x_ptr,
        residual_ptr,
        out_ptr,
        N_ELEMENTS: tl.constexpr,
        SPATIAL: tl.constexpr,
        OUTPUT_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N_ELEMENTS

        gate = tl.load(gate_ptr + offsets // SPATIAL, mask=mask, other=0.0).to(tl.float32)
        gate_scale = tl.sigmoid(gate)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mul_tensor = x * gate_scale
        mul_tensor_1 = mul_tensor * 2.0
        mul_tensor_2 = mul_tensor_1 * 0.2
        add_tensor = residual + mul_tensor_2
        div_tensor = add_tensor / (tl.exp(-add_tensor) + 1.0)
        out = div_tensor * OUTPUT_SCALE
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_ROWS": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_ROWS": 16, "BLOCK_HW": 128}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_ROWS": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=4),
        ],
        key=["CHANNELS", "HEIGHT", "WIDTH", "LAYOUT_TAG"],
    )
    @triton.jit
    def _nfnet_silu_gate_kernel(
        gate_ptr,
        x_ptr,
        residual_ptr,
        out_ptr,
        GATE_STRIDE_N: tl.constexpr,
        GATE_STRIDE_C: tl.constexpr,
        X_STRIDE_N: tl.constexpr,
        X_STRIDE_C: tl.constexpr,
        X_STRIDE_H: tl.constexpr,
        X_STRIDE_W: tl.constexpr,
        RESIDUAL_STRIDE_N: tl.constexpr,
        RESIDUAL_STRIDE_C: tl.constexpr,
        RESIDUAL_STRIDE_H: tl.constexpr,
        RESIDUAL_STRIDE_W: tl.constexpr,
        OUT_STRIDE_N: tl.constexpr,
        OUT_STRIDE_C: tl.constexpr,
        OUT_STRIDE_H: tl.constexpr,
        OUT_STRIDE_W: tl.constexpr,
        BATCH: tl.constexpr,
        CHANNELS: tl.constexpr,
        HEIGHT: tl.constexpr,
        WIDTH: tl.constexpr,
        LAYOUT_TAG: tl.constexpr,
        OUTPUT_SCALE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)

        hw = HEIGHT * WIDTH
        total_rows = BATCH * CHANNELS
        row_mask = row_offsets < total_rows
        hw_mask = hw_offsets < hw
        mask = row_mask[:, None] & hw_mask[None, :]

        n_offsets = row_offsets // CHANNELS
        c_offsets = row_offsets - n_offsets * CHANNELS
        h_offsets = hw_offsets // WIDTH
        w_offsets = hw_offsets - h_offsets * WIDTH

        gate_offsets = n_offsets * GATE_STRIDE_N + c_offsets * GATE_STRIDE_C
        gate = tl.load(gate_ptr + gate_offsets, mask=row_mask, other=0.0).to(tl.float32)
        gate_scale = tl.sigmoid(gate)

        x_offsets = (
            n_offsets[:, None] * X_STRIDE_N
            + c_offsets[:, None] * X_STRIDE_C
            + h_offsets[None, :] * X_STRIDE_H
            + w_offsets[None, :] * X_STRIDE_W
        )
        residual_offsets = (
            n_offsets[:, None] * RESIDUAL_STRIDE_N
            + c_offsets[:, None] * RESIDUAL_STRIDE_C
            + h_offsets[None, :] * RESIDUAL_STRIDE_H
            + w_offsets[None, :] * RESIDUAL_STRIDE_W
        )
        out_offsets = (
            n_offsets[:, None] * OUT_STRIDE_N
            + c_offsets[:, None] * OUT_STRIDE_C
            + h_offsets[None, :] * OUT_STRIDE_H
            + w_offsets[None, :] * OUT_STRIDE_W
        )

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)

        mul_tensor = x * gate_scale[:, None]
        mul_tensor_1 = mul_tensor * 2.0
        mul_tensor_2 = mul_tensor_1 * 0.2
        add_tensor = residual + mul_tensor_2
        div_tensor = add_tensor / (tl.exp(-add_tensor) + 1.0)
        out = div_tensor * OUTPUT_SCALE
        tl.store(out_ptr + out_offsets, out, mask=mask)


def _expect_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if value.ndim != 4:
        raise ValueError(f"{name} must be rank 4, got rank {value.ndim}")
    return value


def _layout_tag(tensor: torch.Tensor) -> int:
    if tensor.is_contiguous():
        return 0
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return 1
    return 2


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_nfnet_silu_gate.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    gate = _expect_tensor("convolution_73", inputs[0])
    x = _expect_tensor("convolution_71", inputs[1])
    residual = _expect_tensor("add_100", inputs[2])

    if tuple(gate.shape[2:]) != (1, 1):
        raise ValueError(f"convolution_73 must have spatial shape (1, 1), got {tuple(gate.shape)}")
    if tuple(x.shape) != tuple(residual.shape):
        raise ValueError(f"convolution_71 shape {tuple(x.shape)} != add_100 shape {tuple(residual.shape)}")
    if tuple(gate.shape[:2]) != tuple(x.shape[:2]):
        raise ValueError(f"gate batch/channel shape {tuple(gate.shape[:2])} != data shape {tuple(x.shape[:2])}")
    if x.device != gate.device or residual.device != gate.device:
        raise ValueError("all inputs must be on the same CUDA device")
    if tuple(x.stride()) != tuple(residual.stride()):
        raise ValueError(
            f"convolution_71 stride {tuple(x.stride())} != add_100 stride {tuple(residual.stride())}"
        )
    if any(t.storage_offset() != 0 for t in (gate, x, residual)):
        raise ValueError("captured repro inputs are expected to have storage_offset 0")
    return gate, x, residual


@oracle_impl(hardware="H100", shapes="(T([128, 1536, 1, 1], f32), T([128, 1536, 7, 7], f32), T([128, 1536, 7, 7], f32))")
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
    gate, x, residual = _validate_inputs(inputs)
    batch, channels, height, width = x.shape
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )

    if gate.is_contiguous() and x.is_contiguous() and residual.is_contiguous() and output.is_contiguous():
        n_elements = x.numel()
        spatial = height * width
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_N"]),)
        _nfnet_silu_gate_flat_kernel[grid](
            gate,
            x,
            residual,
            output,
            N_ELEMENTS=n_elements,
            SPATIAL=spatial,
            OUTPUT_SCALE=FINAL_SCALE,
        )
        return output

    grid = lambda meta: (
        triton.cdiv(batch * channels, meta["BLOCK_ROWS"]),
        triton.cdiv(height * width, meta["BLOCK_HW"]),
    )
    _nfnet_silu_gate_kernel[grid](
        gate,
        x,
        residual,
        output,
        GATE_STRIDE_N=gate.stride(0),
        GATE_STRIDE_C=gate.stride(1),
        X_STRIDE_N=x.stride(0),
        X_STRIDE_C=x.stride(1),
        X_STRIDE_H=x.stride(2),
        X_STRIDE_W=x.stride(3),
        RESIDUAL_STRIDE_N=residual.stride(0),
        RESIDUAL_STRIDE_C=residual.stride(1),
        RESIDUAL_STRIDE_H=residual.stride(2),
        RESIDUAL_STRIDE_W=residual.stride(3),
        OUT_STRIDE_N=output.stride(0),
        OUT_STRIDE_C=output.stride(1),
        OUT_STRIDE_H=output.stride(2),
        OUT_STRIDE_W=output.stride(3),
        BATCH=batch,
        CHANNELS=channels,
        HEIGHT=height,
        WIDTH=width,
        LAYOUT_TAG=_layout_tag(x),
        OUTPUT_SCALE=FINAL_SCALE,
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
