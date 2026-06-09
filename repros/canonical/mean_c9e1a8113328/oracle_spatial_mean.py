"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete NFNet spatial mean scope in one Triton row-reduction kernel, reading each `[H,W]` tile once and writing the final keepdim `[N,C,1,1]` contiguous result directly, whereas Inductor lowers this standalone `aten.mean.dim(..., [2, 3], keepdim=True)` through a much slower generic reduction schedule on the captured `[128,1536,6,6]` shape; Inductor cannot do this today because its norm-template/reduction scheduler does not recognize a producer-free fixed-small-spatial mean as a direct `(N,C)` row reduction with simple output-layout stores; the fix is NEW_PATTERN: add a guarded small-spatial keepdim-mean reduction template or schedule rule that selects this row-tiled lowering for NCHW tensors when benchmarking shows it beats the generic reduction path."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
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
    def _spatial_mean_kernel(
        x_ptr,
        out_ptr,
        stride_n: tl.constexpr,
        stride_c: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw_offsets = tl.arange(0, BLOCK_HW)

        n_offsets = row_offsets // channels
        c_offsets = row_offsets - n_offsets * channels
        h_offsets = hw_offsets // width
        w_offsets = hw_offsets - h_offsets * width

        valid_rows = row_offsets < total_rows
        valid_hw = hw_offsets < hw
        x_offsets = (
            n_offsets[:, None] * stride_n
            + c_offsets[:, None] * stride_c
            + h_offsets[None, :] * stride_h
            + w_offsets[None, :] * stride_w
        )
        values = tl.load(
            x_ptr + x_offsets,
            mask=valid_rows[:, None] & valid_hw[None, :],
            other=0.0,
        ).to(tl.float32)
        reduced = tl.sum(values, axis=1) * (1.0 / hw)
        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


def _require_input(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")

    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input must be a tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"{REPRO_ID} expects a 4D NCHW tensor, got shape {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"{REPRO_ID} expects f16/f32 input, got {x.dtype}")
    if not x.is_cuda:
        raise RuntimeError(f"{REPRO_ID} expects CUDA input")
    if x.shape[2] < 1 or x.shape[3] < 1:
        raise ValueError(f"{REPRO_ID} expects non-empty spatial dims, got {tuple(x.shape)}")
    return x


def _output_for(x: torch.Tensor) -> torch.Tensor:
    n, c, _, _ = (int(dim) for dim in x.shape)
    return torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=x.device,
        dtype=x.dtype,
    )


def _kernel_params(x: torch.Tensor) -> tuple[int, int]:
    hw = int(x.shape[2]) * int(x.shape[3])
    block_hw = triton.next_power_of_2(hw)
    if block_hw > 4096:
        raise ValueError(f"{REPRO_ID} spatial tile is too large for this oracle: {hw}")
    if hw <= 64:
        block_rows = 64
    elif hw <= 256:
        block_rows = 16
    elif hw <= 1024:
        block_rows = 8
    else:
        block_rows = 2
    return block_rows, block_hw


@oracle_impl(hardware="H100", shapes="(T([128, 1536, 6, 6], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete `aten.mean.dim(..., [2, 3], keepdim=True)` repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_spatial_mean.py")

    x = _require_input(inputs)
    output = _output_for(x)
    n, c, h, w = (int(dim) for dim in x.shape)
    total_rows = n * c
    hw = h * w
    block_rows, block_hw = _kernel_params(x)
    _spatial_mean_kernel[(triton.cdiv(total_rows, block_rows),)](
        x,
        output,
        stride_n=x.stride(0),
        stride_c=x.stride(1),
        stride_h=x.stride(2),
        stride_w=x.stride(3),
        total_rows=total_rows,
        channels=c,
        width=w,
        hw=hw,
        BLOCK_ROWS=block_rows,
        BLOCK_HW=block_hw,
        num_warps=4,
        num_stages=3,
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
