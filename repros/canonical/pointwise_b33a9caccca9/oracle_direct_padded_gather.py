"""Gap diagnosis (classification: NEW_PATTERN): this oracle writes the indexed relative-position table directly into the padded backing layout and returns the final expanded view, whereas Inductor lowers the gather, layout conversion, pad, slice, and expand as separate scheduling concerns; Inductor cannot do this today because it lacks a pattern that recognizes pad-then-slice as a layout request on top of an indexed gather and emits the destination-strided gather directly; the fix is NEW_PATTERN: add a gather-to-padded-layout lowering that preserves the final view metadata while avoiding intermediate materialization."""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _padded_gather_all_channels_kernel(
        index_ptr,
        table_ptr,
        out_ptr,
        index_s0: tl.constexpr,
        index_s1: tl.constexpr,
        table_s0: tl.constexpr,
        table_s1: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        C: tl.constexpr,
        PADDED_W: tl.constexpr,
        VISIBLE_BLOCKS: tl.constexpr,
        PAD_COLS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < VISIBLE_BLOCKS:
            spatial = pid * BLOCK_M + tl.arange(0, BLOCK_M)
            row = spatial // W
            col = spatial - row * W
            spatial_mask = spatial < (H * W)

            indices = tl.load(
                index_ptr + row * index_s0 + col * index_s1,
                mask=spatial_mask,
                other=0,
            )
            channels = tl.arange(0, BLOCK_C)
            values = tl.load(
                table_ptr + indices[:, None] * table_s0 + channels[None, :] * table_s1,
                mask=spatial_mask[:, None] & (channels[None, :] < C),
                other=0.0,
            )
            out_offsets = (
                channels[None, :] * H * PADDED_W
                + row[:, None] * PADDED_W
                + col[:, None]
            )
            tl.store(
                out_ptr + out_offsets,
                values,
                mask=spatial_mask[:, None] & (channels[None, :] < C),
            )
        else:
            pad_offsets = (pid - VISIBLE_BLOCKS) * BLOCK_PAD + tl.arange(0, BLOCK_PAD)
            pad_mask = pad_offsets < (C * H * PAD_COLS)
            zero_col = W + (pad_offsets % PAD_COLS)
            zero_row_channel = pad_offsets // PAD_COLS
            zero_row = zero_row_channel % H
            zero_channel = zero_row_channel // H
            tl.store(
                out_ptr + zero_channel * H * PADDED_W + zero_row * PADDED_W + zero_col,
                tl.zeros((BLOCK_PAD,), tl.float32),
                mask=pad_mask,
            )


@oracle_impl(hardware="H100", shapes="(T([197, 197], i64, gen=Index(732)), T([732, 12], f32), S([197, 197, -1]), S([128, 12, 197, 197]))")
def oracle_forward(inputs):
    """Run the full Repro computation with direct destination-strided gather."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    index, table, _shape_param_0, output_shape = inputs
    height = index.shape[0]
    width = index.shape[1]
    channels = table.shape[1]
    padded_width = width + 3

    backing = torch.empty(
        (1, channels, height, padded_width),
        device=table.device,
        dtype=table.dtype,
    )
    visible_blocks = triton.cdiv(height * width, 64)
    pad_blocks = triton.cdiv(channels * height * 3, 256)
    grid = (visible_blocks + pad_blocks,)
    _padded_gather_all_channels_kernel[grid](
        index,
        table,
        backing,
        index.stride(0),
        index.stride(1),
        table.stride(0),
        table.stride(1),
        height,
        width,
        channels,
        padded_width,
        visible_blocks,
        3,
        BLOCK_M=64,
        BLOCK_C=16,
        BLOCK_PAD=256,
    )

    return backing.as_strided(
        tuple(output_shape),
        (0, height * padded_width, padded_width, 1),
        0,
    )


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
