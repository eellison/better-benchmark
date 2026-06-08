"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full NFNet training weight-standardization scope in one fixed-shape two-warp Triton kernel, matching Inductor's Welford population `var_mean` over each `[1,3072,1536]` logical row, `libdevice.rsqrt(var + 1e-5)`, the generated multiply order `((x - mean) * invstd) * (gain * 0.02551551815399144)`, and the returned invstd, contiguous standardized weight, and mean view; Inductor already fuses the scope into one generated reduction kernel, but its generic reduction scheduling uses heavier wrapper/tile choices for this fixed 1536-wide weight-standardization row; Inductor cannot select this today because it lacks a dedicated exact-numerics weight-standardization row-reduction pattern with side-output-aware direct stores and tuned low-warp Welford lowering; the fix is NEW_PATTERN: add a specialized weight-standardization/norm-template lowering for fixed conv-weight rows that preserves Welford accumulation and the same rounding boundaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice
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

OUT_CHANNELS = 3072
IN_CHANNELS = 1536
KERNEL_SIZE = 1
KERNEL_ELEMS = KERNEL_SIZE * KERNEL_SIZE
ROW_ELEMS = IN_CHANNELS * KERNEL_ELEMS
EPS = 1.0e-5
SCALE = 0.02551551815399144
BLOCK_ROW = 2048

if triton is not None:

    @triton.jit
    def _weight_standardization_kernel(
        weight_ptr,
        gain_ptr,
        invstd_ptr,
        standardized_ptr,
        mean_view_ptr,
        row_elems: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
    ):
        oc = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_ROW)[None, :]

        mean_acc = tl.zeros([1, BLOCK_ROW], tl.float32)
        m2_acc = tl.zeros([1, BLOCK_ROW], tl.float32)
        weight_acc = tl.zeros([1, BLOCK_ROW], tl.float32)
        for row_offset in tl.range(0, row_elems, BLOCK_ROW):
            r = row_offset + offsets
            row_mask = r < row_elems
            x = tl.load(
                weight_ptr + oc * row_elems + r,
                mask=row_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
                x,
                mean_acc,
                m2_acc,
                weight_acc,
                row_offset == 0,
            )
            mean_acc = tl.where(row_mask, mean_next, mean_acc)
            m2_acc = tl.where(row_mask, m2_next, m2_acc)
            weight_acc = tl.where(row_mask, weight_next, weight_acc)

        mean_row, m2_row, _ = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
        invstd_row = libdevice.rsqrt((m2_row / row_elems) + eps)
        mean = mean_row[:, None]
        invstd = invstd_row[:, None]
        gain_scaled = tl.load(gain_ptr + oc, eviction_policy="evict_last") * scale
        one = tl.arange(0, 1)

        tl.store(invstd_ptr + oc + one, invstd_row, mask=one == 0)
        tl.store(mean_view_ptr + oc + one, mean_row, mask=one == 0)

        for row_offset in tl.range(0, row_elems, BLOCK_ROW):
            r = row_offset + offsets
            row_mask = r < row_elems
            x = tl.load(
                weight_ptr + oc * row_elems + r,
                mask=row_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            centered = x - mean
            normalized = centered * invstd
            out = normalized * gain_scaled
            tl.store(standardized_ptr + oc * row_elems + r, out, mask=row_mask)


def _require_tensor(
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
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    weight, gain, reshape_shape, output_shape = inputs
    weight_t = _require_tensor(
        "primals_230",
        weight,
        (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE),
        (ROW_ELEMS, KERNEL_ELEMS, KERNEL_SIZE, 1),
    )
    gain_t = _require_tensor(
        "primals_231",
        gain,
        (OUT_CHANNELS, 1, 1, 1),
        (1, 1, 1, 1),
    )

    if list(reshape_shape) not in ([1, OUT_CHANNELS, ROW_ELEMS], [1, OUT_CHANNELS, -1]):
        raise ValueError(f"unexpected reshape shape parameter: {reshape_shape!r}")
    if list(output_shape) != [OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE]:
        raise ValueError(f"unexpected output shape parameter: {output_shape!r}")
    if gain_t.device != weight_t.device:
        raise ValueError("weight and gain must be on the same CUDA device")

    return weight_t, gain_t


def oracle_forward(inputs):
    """Run the full NFNet training weight-standardization repro computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_weight_standardization.py")

    weight, gain = _validate_inputs(inputs)
    invstd = torch.empty_strided(
        (OUT_CHANNELS,),
        (1,),
        device=weight.device,
        dtype=torch.float32,
    )
    standardized = torch.empty_strided(
        (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE),
        (ROW_ELEMS, KERNEL_ELEMS, KERNEL_SIZE, 1),
        device=weight.device,
        dtype=torch.float32,
    )
    mean_view = torch.empty_strided(
        (1, OUT_CHANNELS, 1),
        (OUT_CHANNELS, 1, 1),
        device=weight.device,
        dtype=torch.float32,
    )

    _weight_standardization_kernel[(OUT_CHANNELS,)](
        weight,
        gain,
        invstd,
        standardized,
        mean_view,
        row_elems=ROW_ELEMS,
        eps=EPS,
        scale=SCALE,
        BLOCK_ROW=BLOCK_ROW,
        num_warps=2,
        num_stages=3,
    )
    return (invstd, standardized, mean_view)


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
