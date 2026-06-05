"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete CycleGAN instance-normalization plus residual-add scope in one shape-specialized Triton channel kernel, including fp32 population `var_mean` over each `[64,64]` plane, eps=1e-5 `rsqrt`, and the final contiguous `[1,256,64,64]` output while reusing the loaded activation values for the epilogue, whereas Inductor's norm-template lowering still leaves enough generic reduction/epilogue scheduling overhead to run slower on this fixed N=1 spatial instance-norm case; Inductor cannot do this exact floor today because the scheduler does not select a specialized one-program-per-channel plan that keeps the reduction tile live through the residual-add output store; the fix is SCHEDULER_FUSION: add a guarded fixed-spatial instance-norm residual template that fuses the reduction and residual epilogue into the same channel-tiled program."""
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
    has_stochastic_ops,
)


BATCH = 1
CHANNELS = 256
HEIGHT = 64
WIDTH = 64
HW = HEIGHT * WIDTH
TOTAL_ELEMENTS = BATCH * CHANNELS * HW
EPS = 1.0e-5
BLOCK_HW = 4096
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _instance_norm_residual_kernel(
        x_ptr,
        residual_ptr,
        out_ptr,
        hw: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_M)
        mask = offsets < hw
        channel_offsets = channel * hw + offsets

        x = tl.load(x_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / hw
        centered = x - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hw
        invstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)

        residual = tl.load(residual_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        out = residual + centered * invstd
        tl.store(out_ptr + channel_offsets, out, mask=mask)


def _expect_f32_contiguous(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x = _expect_f32_contiguous("convolution_20", inputs[0], (BATCH, CHANNELS, HEIGHT, WIDTH))
    residual = _expect_f32_contiguous("add_26", inputs[1], (BATCH, CHANNELS, HEIGHT, WIDTH))
    if residual.device != x.device:
        raise ValueError("all tensor inputs must be on the same device")
    return x, residual


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x, residual = _validate_inputs(inputs)
    var, mean = torch.ops.aten.var_mean.correction(
        x,
        [0, 2, 3],
        correction=0,
        keepdim=True,
    )
    return residual + (x - mean) * torch.rsqrt(var + EPS)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: accepts the same two inputs as Repro.forward() and returns
    the same single f32 `[1,256,64,64]` contiguous tensor.
    """
    x, residual = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    out = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
        device=x.device,
        dtype=torch.float32,
    )
    _instance_norm_residual_kernel[(CHANNELS,)](
        x,
        residual,
        out,
        hw=HW,
        eps=EPS,
        BLOCK_M=BLOCK_HW,
        num_warps=8,
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
