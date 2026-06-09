"""
Full-scope oracle for sum_sum_sum_5c7c5e63becb (Blenderbot LN/dropout backward).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle consumes the
same inputs as repro.py and returns the same four outputs, including the two
source column reductions, the materialized dropout-scaled layer-norm backward
side output returned as a [2560, 2048] transpose view, and the downstream
column reduction over that side output. It differs from Inductor by doing the
row-local layer-norm reductions, side-output store, and sibling column
accumulators in one row-split Triton producer, then finalizing the column
partials; Inductor cannot do this today because its scheduler represents the
hidden-dimension row reductions, batch/sequence column reductions, dropout
epilogue, view/permute side output, and dependent final reduction as separate
generic nodes rather than one coordinated dependent multi-output reduction.
The fix is COOPERATIVE_SPLIT_K support for dependent multi-output reductions
with materialized side outputs, so row tiles can share row scalar accumulators
and cooperatively reduce all compatible column outputs without rereading the
side-output buffer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

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


REPRO_ID = "sum_sum_sum_5c7c5e63becb"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ = 128
ROWS = BATCH * SEQ
CHANNELS = 2560
DROPOUT_SCALE = 1.1111111111111112
HISTORICAL_BEST_COMPILE_US = 38.72000053524971
ROW_SPLIT = 16
FINALIZE_BLOCK_C = 32


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


@triton.jit
def _row_block_full_scope_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    residual_ptr,
    keep_ptr,
    out_base_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_C], dtype=tl.float32)

    for local_row in tl.range(0, ROW_SPLIT):
        row = row_block * ROW_SPLIT + local_row
        row_mask = row < ROWS_
        mask = row_mask & col_mask
        offsets = row * CHANNELS_ + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        gate = tl.load(gate_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=0)

        out = residual + gate * (weighted * CHANNELS_ - row_sum - rhs * row_dot)
        out = out * keep * DROPOUT_SCALE_

        tl.store(out_base_ptr + offsets, out, mask=mask)
        acc_x_rhs += tl.where(mask, x * rhs, 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_out += tl.where(mask, out, 0.0)

    partial_offsets = row_block * CHANNELS_ + cols
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)
    tl.store(partial_out_ptr + partial_offsets, acc_out, mask=col_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_sum_out_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    col_mask = cols < CHANNELS_
    block_mask = blocks < NUM_ROW_BLOCKS
    mask = block_mask[:, None] & col_mask[None, :]
    offsets = blocks[:, None] * CHANNELS_ + cols[None, :]

    x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_sum_out_ptr + cols, tl.sum(out, axis=0), mask=col_mask)


def oracle_fused(
    mm_2: torch.Tensor,
    arg11_1: torch.Tensor,
    arg34_1: torch.Tensor,
    arg39_1: torch.Tensor,
    arg43_1: torch.Tensor,
    arg33_1: torch.Tensor,
    *_shape_params: object,
    row_split: int = ROW_SPLIT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_2.shape == (ROWS, CHANNELS)
    assert arg11_1.shape == (CHANNELS,)
    assert arg34_1.shape == (BATCH, SEQ, CHANNELS)
    assert arg39_1.shape == (BATCH, SEQ, 1)
    assert arg43_1.shape == (BATCH, SEQ, CHANNELS)
    assert arg33_1.shape == (BATCH, SEQ, CHANNELS)

    rhs = arg34_1.reshape(ROWS, CHANNELS)
    gate = arg39_1.reshape(ROWS)
    residual = arg43_1.reshape(ROWS, CHANNELS)
    keep = arg33_1.reshape(ROWS, CHANNELS)
    device = mm_2.device

    num_row_blocks = triton.cdiv(ROWS, row_split)
    partial_x_rhs = torch.empty((num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    partial_out = torch.empty((num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    out_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    _row_block_full_scope_kernel[(num_row_blocks,)](
        mm_2,
        arg11_1,
        rhs,
        gate,
        residual,
        keep,
        out_base,
        partial_x_rhs,
        partial_x,
        partial_out,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        ROW_SPLIT=row_split,
        BLOCK_C=4096,
        num_warps=16,
    )

    out_x_rhs = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(CHANNELS, FINALIZE_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_out,
        out_x_rhs,
        out_x,
        out_sum_out,
        NUM_ROW_BLOCKS=num_row_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=FINALIZE_BLOCK_C,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_base.permute(1, 0), out_sum_out


@oracle_impl(hardware="H100", shapes="(T([2048, 2560], f32), T([2560], f32), T([16, 128, 2560], f32), T([16, 128, 1], f32), T([16, 128, 2560], f32), T([16, 128, 2560], b8), S([16, 128, 2560]), S([2048, 2560]), S([2560]))")
def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    """Same inputs and outputs as Repro.forward; no subset timing."""
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def main() -> None:
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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
