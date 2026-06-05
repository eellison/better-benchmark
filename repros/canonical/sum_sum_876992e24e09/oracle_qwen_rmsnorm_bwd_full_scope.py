"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Qwen/Llama/Mistral RMSNorm-backward fragment by row-tiling the `[2048, 1024]` producer, sharing each row's hidden-dimension dot product for the transposed bf16 input-gradient output while accumulating cooperative per-channel partial sums for the sibling bf16 `[1024]` weight-gradient reduction, whereas Inductor currently schedules the bf16 adds, row reduction, column reduction, casts, and transposed side output as separate generic reduction/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that keeps bf16-rounded row-local values live while also materializing a layout-changing side output and finalizing column partials; the fix is COOPERATIVE_SPLIT_K: add a row-tiled multi-output RMSNorm-backward reduction lowering that fuses the bf16 producers, row-summary epilogue, transposed stores, and channel-gradient finalization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 2048
CHANNELS = 1024
ROW_TILE = 16
BLOCK_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(shape_param) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape_param)


if triton is not None:

    @triton.jit
    def _bf16_add(lhs, rhs):
        return (lhs + rhs).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _row_tile_kernel(
        mm_383_ptr,
        mm_385_ptr,
        gamma_ptr,
        arg327_ptr,
        arg312_ptr,
        inv_rms_ptr,
        add_407_ptr,
        out_base_ptr,
        partial_sum_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        rows = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_CHANNELS_)
        row_mask = rows < ROWS_
        col_mask = cols < CHANNELS_
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * CHANNELS_ + cols[None, :]

        mm_383 = tl.load(mm_383_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mm_385 = tl.load(mm_385_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        arg327 = tl.load(arg327_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg312 = tl.load(arg312_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inv_rms = tl.load(inv_rms_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        add_407 = tl.load(add_407_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        upstream = _bf16_add(mm_383, mm_385)
        residual = _bf16_add(arg312, arg327)
        weighted = (upstream * gamma[None, :]).to(tl.bfloat16).to(tl.float32)
        scaled_residual = (
            residual * inv_rms[:, None]
        ).to(tl.bfloat16).to(tl.float32)

        weight_grad_product = (
            upstream * scaled_residual
        ).to(tl.bfloat16).to(tl.float32)
        row_dot = tl.sum(tl.where(mask, weighted * residual, 0.0), axis=1)

        inv_rms2 = inv_rms * inv_rms
        correction = (
            (-0.5 * row_dot * inv_rms2 * inv_rms) / CHANNELS_
        )[:, None] * (2.0 * residual)
        input_grad = weighted * inv_rms[:, None] + correction
        input_grad_bf16 = input_grad.to(tl.bfloat16).to(tl.float32)
        out = (add_407 + input_grad_bf16).to(tl.bfloat16)

        tl.store(out_base_ptr + offsets, out, mask=mask)

        partial_offsets = row_tile * CHANNELS_ + cols
        tl.store(
            partial_sum_ptr + partial_offsets,
            tl.sum(tl.where(mask, weight_grad_product, 0.0), axis=0),
            mask=col_mask,
        )

    @triton.jit
    def _finalize_partials_kernel(
        partial_sum_ptr,
        out_sum_ptr,
        NUM_ROW_TILES: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = (tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < CHANNELS_)
        offsets = tiles[:, None] * CHANNELS_ + cols[None, :]
        values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_sum_ptr + cols, tl.sum(values, axis=0), mask=cols < CHANNELS_)


def _validate_inputs(inputs) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 13:
        raise ValueError(f"expected 13 Repro.forward inputs, got {len(inputs)}")

    (
        mm_383,
        mm_385,
        arg10_1,
        arg327_1,
        arg312_1,
        arg328_1,
        add_407,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
        shape_param_4,
        shape_param_5,
    ) = inputs

    expected = {
        "mm_383": (mm_383, (ROWS, CHANNELS), torch.bfloat16),
        "mm_385": (mm_385, (ROWS, CHANNELS), torch.bfloat16),
        "arg10_1": (arg10_1, (CHANNELS,), torch.bfloat16),
        "arg327_1": (arg327_1, (ROWS, CHANNELS), torch.bfloat16),
        "arg312_1": (arg312_1, (4, 512, CHANNELS), torch.bfloat16),
        "arg328_1": (arg328_1, (4, 512, 1), torch.float32),
        "add_407": (add_407, (4, 512, CHANNELS), torch.bfloat16),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected {dtype} shape={list(shape)}, "
                f"got dtype={tensor.dtype} shape={list(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the canonical oracle")

    if _shape_tuple(shape_param_0) != (4, 512, CHANNELS):
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0}")
    if _shape_tuple(shape_param_1) != (4, 512, CHANNELS):
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1}")
    if _shape_tuple(shape_param_2) != (4, 512, CHANNELS):
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2}")
    if _shape_tuple(shape_param_3) != (CHANNELS,):
        raise ValueError(f"unexpected _shape_param_3: {shape_param_3}")
    if _shape_tuple(shape_param_4) != (4, 512, CHANNELS):
        raise ValueError(f"unexpected _shape_param_4: {shape_param_4}")
    if _shape_tuple(shape_param_5) != (ROWS, CHANNELS):
        raise ValueError(f"unexpected _shape_param_5: {shape_param_5}")


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    _validate_inputs(inputs)
    (
        mm_383,
        mm_385,
        arg10_1,
        arg327_1,
        arg312_1,
        arg328_1,
        add_407,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        shape_param_3,
        _shape_param_4,
        _shape_param_5,
    ) = inputs

    num_row_tiles = triton.cdiv(ROWS, ROW_TILE)
    out_sum = torch.empty(
        _shape_tuple(shape_param_3),
        device=mm_383.device,
        dtype=torch.bfloat16,
    )
    out_base = torch.empty(
        (ROWS, CHANNELS),
        device=mm_383.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty(
        (num_row_tiles, CHANNELS),
        device=mm_383.device,
        dtype=torch.float32,
    )

    _row_tile_kernel[(num_row_tiles,)](
        mm_383,
        mm_385,
        arg10_1,
        arg327_1,
        arg312_1,
        arg328_1,
        add_407,
        out_base,
        partial_sum,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        BLOCK_ROWS=ROW_TILE,
        BLOCK_CHANNELS_=BLOCK_CHANNELS,
        num_warps=8,
    )

    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_sum,
        out_sum,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=num_row_tiles,
        BLOCK_CHANNELS_=FINAL_BLOCK_CHANNELS,
        num_warps=4,
    )

    return out_sum, out_base.permute(1, 0)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
