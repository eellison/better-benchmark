"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Swin window-reverse layernorm-backward scope, including the fixed 7x7 window reverse clone, both row-local fp32 hidden-size-256 reductions, the residual gradient side output returned as the required `[256,100352]` transpose view, and the two sibling `[256]` channel reductions by row-tiling the K dimension and atomically accumulating per-channel partials from the same Triton pass, whereas Inductor currently materializes/schedules the layout clone, row reductions, channel reductions, and transpose-producing epilogue as separate generic regions; Inductor cannot do this today because the scheduler has no representation for a dependent multi-output reduction that keeps row-wise scalar intermediates in registers while cooperatively split-K accumulating compatible channel reductions and writing a layout side output; the fix is COOPERATIVE_SPLIT_K: add a multi-output reduction template that can split the row dimension, accumulate per-channel partials, and sink deterministic layout remaps plus side-output stores into the same fused plan."""
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _zero_pair_kernel(
        out0_ptr,
        out1_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        zeros = tl.zeros((BLOCK,), dtype=tl.float32)
        tl.store(out0_ptr + offsets, zeros, mask=mask)
        tl.store(out1_ptr + offsets, zeros, mask=mask)

    @triton.jit
    def _atomic_swin_ln_backward_kernel(
        mm_183_ptr,
        gamma_ptr,
        mm_ptr,
        mean_ptr,
        rsqrt_ptr,
        residual_ptr,
        out_base_ptr,
        out_xhat_ptr,
        out_x_ptr,
        ROWS_: tl.constexpr,
        C_: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
        C_BLOCK: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        c_offsets = tl.arange(0, C_BLOCK)
        row_mask = row_offsets < ROWS_
        c_mask = c_offsets < C_
        mask = row_mask[:, None] & c_mask[None, :]

        batch = row_offsets // 784
        spatial = row_offsets - batch * 784
        h = spatial // 28
        w = spatial - h * 28
        h_block = h // 7
        w_block = w // 7
        h_inner = h - h_block * 7
        w_inner = w - w_block * 7
        source_rows = (
            batch * 784
            + h_block * 196
            + w_block * 49
            + h_inner * 7
            + w_inner
        )

        mm_183_offsets = source_rows[:, None] * C_ + c_offsets[None, :]
        row_offsets_2d = row_offsets[:, None] * C_ + c_offsets[None, :]
        gamma = tl.load(gamma_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        x = tl.load(mm_183_ptr + mm_183_offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(mm_ptr + row_offsets_2d, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
        rsqrt = tl.load(rsqrt_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row_offsets_2d, mask=mask, other=0.0).to(tl.float32)

        mul_tensor = x * gamma[None, :]
        mul_tensor_1 = mul_tensor * 256.0
        sub_tensor = grad - mean[:, None]
        mul_tensor_2 = sub_tensor * rsqrt[:, None]
        mul_tensor_3 = mul_tensor * mul_tensor_2
        sum_dim_int_list = tl.sum(tl.where(mask, mul_tensor, 0.0), axis=1)
        sum_dim_int_list_1 = tl.sum(tl.where(mask, mul_tensor_3, 0.0), axis=1)
        mul_tensor_4 = mul_tensor_2 * sum_dim_int_list_1[:, None]
        sub_tensor_1 = mul_tensor_1 - sum_dim_int_list[:, None]
        sub_tensor_2 = sub_tensor_1 - mul_tensor_4
        div_tensor = rsqrt / 256.0
        mul_tensor_5 = div_tensor[:, None] * sub_tensor_2
        add_tensor = residual + mul_tensor_5
        tl.store(out_base_ptr + row_offsets_2d, add_tensor, mask=mask)

        mul_tensor_6 = x * mul_tensor_2
        acc_xhat = tl.sum(tl.where(mask, mul_tensor_6, 0.0), axis=0)
        acc_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        tl.atomic_add(out_xhat_ptr + c_offsets, acc_xhat, sem="relaxed", mask=c_mask)
        tl.atomic_add(out_x_ptr + c_offsets, acc_x, sem="relaxed", mask=c_mask)

ROWS = 100352
BATCH = 128
HEIGHT = 28
WIDTH = 28
CHANNELS = 256
ATOMIC_ROW_BLOCK = 32
ATOMIC_C_BLOCK = 256
ZERO_BLOCK = 1024
# Numeric choices: all captured values are fp32; the oracle keeps aten operation
# order and the two hidden-size reductions in fp32, with no alternate math.


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} expected shape={shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} expected dtype={dtype}, got {value.dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _require_shape_param(name: str, value: Any, expected: list[int]) -> None:
    if list(value) != expected:
        raise ValueError(f"{name} expected {expected}, got {value}")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 12:
        raise ValueError(f"expected 12 inputs, got {len(inputs)}")

    (
        mm_183,
        primals_38,
        mm,
        getitem_19,
        rsqrt_6,
        view_1324,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    mm_183 = _require_tensor("mm_183", mm_183, (ROWS, CHANNELS), torch.float32)
    primals_38 = _require_tensor("primals_38", primals_38, (CHANNELS,), torch.float32)
    mm = _require_tensor("mm", mm, (ROWS, CHANNELS), torch.float32)
    getitem_19 = _require_tensor("getitem_19", getitem_19, (BATCH, HEIGHT, WIDTH, 1), torch.float32)
    rsqrt_6 = _require_tensor("rsqrt_6", rsqrt_6, (BATCH, HEIGHT, WIDTH, 1), torch.float32)
    view_1324 = _require_tensor(
        "view_1324",
        view_1324,
        (BATCH, HEIGHT, WIDTH, CHANNELS),
        torch.float32,
    )

    _require_shape_param("_shape_param_0", shape0, [2048, 49, CHANNELS])
    _require_shape_param("_shape_param_1", shape1, [2048, 7, 7, CHANNELS])
    _require_shape_param("_shape_param_2", shape2, [BATCH, 4, 4, 7, 7, CHANNELS])
    _require_shape_param("_shape_param_3", shape3, [BATCH, HEIGHT, WIDTH, CHANNELS])
    _require_shape_param("_shape_param_4", shape4, [BATCH, HEIGHT, WIDTH, CHANNELS])
    _require_shape_param("_shape_param_5", shape5, [ROWS, CHANNELS])

    return mm_183, primals_38, mm, getitem_19, rsqrt_6, view_1324


def oracle_forward(inputs):
    """Run the full repro scope and return the same tuple shapes/dtypes/strides."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    mm_183, primals_38, mm, getitem_19, rsqrt_6, view_1324 = _validate_inputs(inputs)
    device = mm_183.device
    num_row_blocks = triton.cdiv(ROWS, ATOMIC_ROW_BLOCK)

    out0 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_base = torch.empty_strided(
        (ROWS, CHANNELS),
        (CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )

    _zero_pair_kernel[(triton.cdiv(CHANNELS, ZERO_BLOCK),)](
        out0,
        out1,
        CHANNELS,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _atomic_swin_ln_backward_kernel[(num_row_blocks,)](
        mm_183,
        primals_38,
        mm,
        getitem_19,
        rsqrt_6,
        view_1324,
        out_base,
        out0,
        out1,
        ROWS_=ROWS,
        C_=CHANNELS,
        ROW_BLOCK=ATOMIC_ROW_BLOCK,
        C_BLOCK=ATOMIC_C_BLOCK,
        num_warps=8,
    )

    return out0, out1, out_base.permute(1, 0)


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
