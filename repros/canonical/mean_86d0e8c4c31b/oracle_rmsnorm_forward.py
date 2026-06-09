"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 RMSNorm forward returned by Repro.forward, including fp32 promotion, per-row mean(square) over the hidden dimension, eps=1e-6 rsqrt, fp32 weight multiply, and final bf16 output in one Triton row-reduction kernel that keeps the row tile live through the epilogue, whereas Inductor lowers the decomposed convert/pow/mean/add/rsqrt/mul/mul/convert graph through its generic reduction scheduling path; Inductor cannot do this today because its scheduler/codegen does not consistently select a guarded fixed-hidden RMSNorm row schedule that retains the normalized input tile and affine epilogue as one specialization; the fix is SCHEDULER_FUSION: add a benchmark-gated RMSNorm forward schedule/template for contiguous bf16 rows with fp32 weights that fuses the reduction and affine bf16 store directly."""
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

EPS = 1.0e-6

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

if triton is not None:

    @triton.jit
    def _rmsnorm_forward_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
        col_offsets = tl.arange(0, block_h)
        masks = (row_offsets[:, None] < rows) & (col_offsets[None, :] < hidden)
        offsets = row_offsets[:, None] * hidden + col_offsets[None, :]

        x = tl.load(x_ptr + offsets, mask=masks, other=0.0).to(tl.float32)
        sum_sq = tl.sum(tl.where(masks, x * x, 0.0), axis=1)
        inv_rms = tl.rsqrt(sum_sq / hidden + eps)
        weight = tl.load(weight_ptr + col_offsets, mask=col_offsets < hidden, other=0.0)
        y = x * inv_rms[:, None] * weight[None, :]
        tl.store(out_ptr + offsets, y, mask=masks)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, weight = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("both repro inputs must be tensors")
    if x.ndim != 2:
        raise ValueError(f"arg0_1 must be rank-2, got shape={tuple(x.shape)}")
    rows, hidden = (int(x.shape[0]), int(x.shape[1]))
    if tuple(weight.shape) != (hidden,):
        raise ValueError(f"arg1_1 shape {tuple(weight.shape)} != {(hidden,)}")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"arg0_1 dtype {x.dtype} != torch.bfloat16")
    if weight.dtype != torch.float32:
        raise TypeError(f"arg1_1 dtype {weight.dtype} != torch.float32")
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if weight.device != x.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if not x.is_contiguous():
        raise ValueError(f"arg0_1 must be contiguous, got stride={x.stride()}")
    if not weight.is_contiguous():
        raise ValueError(f"arg1_1 must be contiguous, got stride={weight.stride()}")
    if rows <= 0 or hidden <= 0:
        raise ValueError(f"unsupported empty shape {tuple(x.shape)}")
    return x, weight


def _block_m_for_hidden(hidden: int) -> int:
    if hidden <= 512:
        return 4
    return 1


@oracle_impl(hardware="H100", shapes="(T([1152000, 512], bf16), T([512], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete RMSNorm forward repro scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same single contiguous bf16 [rows, hidden] tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_rmsnorm_forward.py")

    x, weight = _validate_inputs(inputs)
    rows, hidden = (int(x.shape[0]), int(x.shape[1]))
    out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    block_m = _block_m_for_hidden(hidden)
    block_h = triton.next_power_of_2(hidden)
    num_warps = 8 if hidden >= 2048 else 4
    grid = (triton.cdiv(rows, block_m),)

    _rmsnorm_forward_kernel[grid](
        x,
        weight,
        out,
        rows=rows,
        hidden=hidden,
        eps=EPS,
        block_m=block_m,
        block_h=block_h,
        num_warps=num_warps,
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
