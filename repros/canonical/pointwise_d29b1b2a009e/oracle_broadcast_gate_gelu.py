"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet broadcast gate plus exact-erf GELU scale with a two-stage storage-order Triton schedule that evaluates the `[N,C,1,1]` sigmoid once per `(N,C)` row and reuses it in the full output pass, whereas Inductor lowers the fused pointwise expression over the output element domain; Inductor cannot reuse that broadcast producer today because generic pointwise scheduling lacks broadcast-invariant scalar caching, but the measured full-scope result is still at the compiled floor because the required exact-erf GELU math, activation/residual reads, and output stores dominate; the fix is BANDWIDTH_BOUND: record this as an at-floor case unless broader pointwise math or memory-codegen improvements move both paths."""
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
GELU_INV_SQRT2 = 0.7071067811865476
NFNET_GELU_SCALE = 1.7015043497085571
NFNET_OUTPUT_SCALE = 0.9622504486493761


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1024}, num_warps=8, num_stages=3),
        ],
        key=["rows_total", "channels"],
    )
    @triton.jit
    def _sigmoid_rows_kernel(
        gate_ptr,
        sigmoid_ptr,
        gate_stride_n: tl.constexpr,
        gate_stride_c: tl.constexpr,
        rows_total: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        mask = rows < rows_total
        n = rows // channels
        c = rows - n * channels
        gate_offsets = n * gate_stride_n + c * gate_stride_c
        gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(sigmoid_ptr + rows, tl.sigmoid(gate), mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["total_elements", "channels", "hw", "layout_key"],
    )
    @triton.jit
    def _apply_gate_gelu_kernel(
        x_ptr,
        residual_ptr,
        sigmoid_ptr,
        scale_ptr,
        out_ptr,
        total_elements: tl.constexpr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        layout_key: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        if layout_key == 0:
            row = offsets // hw
        else:
            n = offsets // (channels * hw)
            c = offsets - (offsets // channels) * channels
            row = n * channels + c

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = tl.load(sigmoid_ptr + row, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)

        gated = x * sigmoid
        gated = gated * 2.0
        gated = gated * scale
        gated = gated * 0.2
        add = gated + residual
        gelu = (add * 0.5) * (tl.math.erf(add * 0.7071067811865476) + 1.0)
        out = gelu * 1.7015043497085571
        out = out * 0.9622504486493761
        tl.store(out_ptr + offsets, out, mask=mask)


def _contiguous_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _, channels, height, width = shape
    return (channels * height * width, height * width, width, 1)


def _channels_last_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _, channels, height, width = shape
    return (channels * height * width, 1, width * channels, channels)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_broadcast_gate_gelu.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    gate = _expect_tensor("convolution_73", inputs[0])
    x = _expect_tensor("convolution_71", inputs[1])
    scale = _expect_tensor("arg211_1", inputs[2], ())
    residual = _expect_tensor("add_100", inputs[3])

    if gate.dim() != 4 or x.dim() != 4 or residual.dim() != 4:
        raise ValueError("gate, x, and residual must all be rank-4 tensors")
    batch, channels, height, width = tuple(x.shape)
    expected_gate_shape = (batch, channels, 1, 1)
    if tuple(gate.shape) != expected_gate_shape:
        raise ValueError(f"convolution_73 has shape {tuple(gate.shape)}, expected {expected_gate_shape}")
    if tuple(residual.shape) != tuple(x.shape):
        raise ValueError(f"add_100 has shape {tuple(residual.shape)}, expected {tuple(x.shape)}")
    if x.device != gate.device or residual.device != gate.device or scale.device != gate.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    for name, tensor in (("convolution_73", gate), ("convolution_71", x), ("add_100", residual)):
        if tensor.storage_offset() != 0:
            raise ValueError(f"{name} must have storage_offset 0, got {tensor.storage_offset()}")

    output_shape = (batch, channels, height, width)
    contiguous_stride = _contiguous_stride(output_shape)
    channels_last_stride = _channels_last_stride(output_shape)
    x_stride = tuple(x.stride())
    residual_stride = tuple(residual.stride())
    if x_stride not in (contiguous_stride, channels_last_stride):
        raise ValueError(
            f"convolution_71 has stride {x_stride}, expected one of "
            f"{contiguous_stride} or {channels_last_stride}"
        )
    if residual_stride != x_stride:
        raise ValueError(
            f"add_100 stride {residual_stride} does not match convolution_71 stride {x_stride}"
        )
    if gate.stride(0) != channels or gate.stride(1) != 1:
        raise ValueError(
            f"convolution_73 has unsupported stride {tuple(gate.stride())}; "
            f"expected row-contiguous N/C strides ({channels}, 1, *, *)"
        )

    layout_key = 0 if x_stride == contiguous_stride else 1
    return gate, x, scale, residual, layout_key


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
    gate, x, scale, residual, layout_key = _validate_inputs(inputs)
    batch, channels, height, width = tuple(x.shape)
    rows_total = batch * channels
    hw = height * width
    output = torch.empty_strided(
        tuple(residual.shape),
        tuple(residual.stride()),
        device=residual.device,
        dtype=residual.dtype,
    )
    sigmoid_rows = torch.empty((rows_total,), device=residual.device, dtype=torch.float32)

    sigmoid_grid = lambda meta: (triton.cdiv(rows_total, meta["BLOCK_ROWS"]),)
    _sigmoid_rows_kernel[sigmoid_grid](
        gate,
        sigmoid_rows,
        gate_stride_n=gate.stride(0),
        gate_stride_c=gate.stride(1),
        rows_total=rows_total,
        channels=channels,
    )
    apply_grid = lambda meta: (triton.cdiv(rows_total * hw, meta["BLOCK_SIZE"]),)
    _apply_gate_gelu_kernel[apply_grid](
        x,
        residual,
        sigmoid_rows,
        scale,
        output,
        total_elements=rows_total * hw,
        channels=channels,
        hw=hw,
        layout_key=layout_key,
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
