"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full sigmoid-gated residual SiLU scaling and 2x2 stride-2 avg_pool2d scope in one Triton output-tiled stencil kernel that reads the unfused inputs and writes only the final f32 `[128,1536,7,7]` tensor, whereas Inductor currently materializes the full f32 `[128,1536,14,14]` pointwise result in one kernel and runs a second avg_pool2d kernel over that buffer; Inductor cannot do this today because scheduler fusion does not sink this broadcast-gated pointwise producer through the fixed-window avg_pool2d consumer and generate the pooled value directly from input windows; the fix is SCHEDULER_FUSION: teach fixed-window pooling codegen to fuse broadcast pointwise producer DAGs into a single stencil-plus-epilogue loop nest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


# --- Oracle kernel(s) ---
BATCH = 128
CHANNELS = 1536
HEIGHT = 14
WIDTH = 14
OUT_HEIGHT = 7
OUT_WIDTH = 7
HW = HEIGHT * WIDTH
OUT_HW = OUT_HEIGHT * OUT_WIDTH
ROWS = BATCH * CHANNELS
SCALE = 0.8980265101338745
BLOCK_ROWS = 8
BLOCK_OUT = 64

if triton is not None:

    @triton.jit
    def _sigmoid_silu_avgpool_kernel(
        gate_ptr,
        conv_ptr,
        residual_ptr,
        out_ptr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_OUT_: tl.constexpr,
        scale: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        out_offsets = tl.arange(0, BLOCK_OUT_)
        row_mask = rows < 196608
        out_mask = out_offsets < 49
        mask = row_mask[:, None] & out_mask[None, :]

        oh = out_offsets // 7
        ow = out_offsets - oh * 7
        input_base = rows[:, None] * 196 + oh[None, :] * 28 + ow[None, :] * 2
        output_base = rows[:, None] * 49 + out_offsets[None, :]

        gate_in = tl.load(gate_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp(-gate_in))
        acc = tl.zeros((BLOCK_ROWS_, BLOCK_OUT_), dtype=tl.float32)

        for dh in tl.static_range(2):
            for dw in tl.static_range(2):
                offset = input_base + dh * 14 + dw
                conv = tl.load(conv_ptr + offset, mask=mask, other=0.0).to(tl.float32)
                residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
                value = conv * gate[:, None]
                value = value * 2.0
                value = value * 0.2
                value = value + residual
                silu = value / (tl.exp(-value) + 1.0)
                acc += silu * scale

        tl.store(out_ptr + output_base, acc * 0.25, mask=mask)


def _require_f32_tensor(name, value, shape, stride):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    gate, conv, residual = inputs
    gate = _require_f32_tensor(
        "convolution_60",
        gate,
        (BATCH, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
    )
    conv = _require_f32_tensor(
        "convolution_58",
        conv,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    residual = _require_f32_tensor(
        "add_81",
        residual,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )

    if conv.device != gate.device or residual.device != gate.device:
        raise ValueError("all tensor inputs must be on the same device")
    return gate, conv, residual


def _torch_oracle(gate, conv, residual):
    gate = torch.ops.aten.sigmoid.default(gate)
    value = torch.ops.aten.mul.Tensor(conv, gate)
    value = torch.ops.aten.mul.Tensor(value, 2.0)
    value = torch.ops.aten.mul.Tensor(value, 0.2)
    value = torch.ops.aten.add.Tensor(value, residual)
    denom = torch.ops.aten.add.Tensor(torch.ops.aten.exp.default(torch.ops.aten.neg.default(value)), 1)
    value = torch.ops.aten.div.Tensor(value, denom)
    value = torch.ops.aten.mul.Tensor(value, SCALE)
    return torch.ops.aten.avg_pool2d.default(value, [2, 2], [2, 2], [0, 0], True, False)


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
    gate, conv, residual = _validate_inputs(inputs)
    if not gate.is_cuda:
        return _torch_oracle(gate, conv, residual)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    out = torch.empty_strided(
        (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1),
        device=gate.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(ROWS, BLOCK_ROWS),)
    _sigmoid_silu_avgpool_kernel[grid](
        gate,
        conv,
        residual,
        out,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_OUT_=BLOCK_OUT,
        scale=SCALE,
        num_warps=4,
        num_stages=3,
    )
    return out


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
