"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete 3-output graph by streaming the added-and-masked producer once per `(batch, two-channel group)`, keeping `sum(where)` and `sum(where * arg52_1)` over each 8x8 spatial block in registers, and deriving the BN-backward-style full tensor epilogue plus both returned channel vectors from those summaries, whereas Inductor currently schedules the add/le/where producer, sibling spatial reductions, dependent grouped reductions, full tensor epilogue, and final channel reductions as separate generic regions over materialized intermediates; Inductor cannot do this today because its algebraic simplifier does not recognize this view-threaded dependent reduction chain or preserve the two base spatial summaries across all consumers; the fix is ALGEBRAIC_ELIMINATION: add a guarded rewrite for this grouped BN-backward-style pattern that lowers the shared spatial summaries and dependent tensor/vector epilogues as one fused multi-output reduction plan."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 64
GROUPS = 32
CHANNELS_PER_GROUP = 2
H = 8
W = 8
HW = H * W
GROUP_SCALE = 0.0078125


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_vectors_kernel(
        out_vec,
        out_sum,
        N_COLS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N_COLS
        zeros = tl.zeros((BLOCK,), dtype=tl.float32)
        tl.store(out_vec + offsets, zeros, mask=mask)
        tl.store(out_sum + offsets, zeros, mask=mask)

    @triton.jit
    def _grouped_spatial_kernel(
        where_12_ptr,
        getitem_48_ptr,
        arg55_ptr,
        full_ptr,
        arg52_ptr,
        arg7_ptr,
        arg54_ptr,
        arg53_ptr,
        out_full_ptr,
        out_vec_ptr,
        out_sum_ptr,
        GROUPS_: tl.constexpr,
        C_: tl.constexpr,
        CHANNELS_PER_GROUP_: tl.constexpr,
        HW_: tl.constexpr,
        GROUP_SCALE_: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid // GROUPS_
        g = pid - n * GROUPS_

        r = tl.arange(0, BLOCK_C)
        hw = tl.arange(0, BLOCK_HW)
        channels = g * CHANNELS_PER_GROUP_ + r
        offsets = n * C_ * HW_ + channels[:, None] * HW_ + hw[None, :]

        add_value = tl.load(where_12_ptr + offsets) + tl.load(getitem_48_ptr + offsets)
        pred = tl.load(arg55_ptr + offsets) <= 0.0
        full_value = tl.load(full_ptr)
        where_value = tl.where(pred, full_value, add_value)
        arg52_value = tl.load(arg52_ptr + offsets)

        sum_where = tl.sum(where_value, axis=1)
        sum_mul = tl.sum(where_value * arg52_value, axis=1)

        gamma = tl.load(arg7_ptr + channels)
        p = tl.load(arg54_ptr + n * GROUPS_ + g)
        q = tl.load(arg53_ptr + n * GROUPS_ + g)

        grouped_mul = tl.sum(sum_mul * gamma, axis=0)
        grouped_where = tl.sum(sum_where * gamma, axis=0)

        p2 = p * p
        m = (grouped_where * q - grouped_mul) * p2 * p * GROUP_SCALE_
        bias = -m * q - grouped_where * p * GROUP_SCALE_
        out = where_value * (p * gamma)[:, None] + arg52_value * m + bias

        tl.store(out_full_ptr + offsets, out)
        tl.atomic_add(out_vec_ptr + channels, (sum_mul - sum_where * q) * p, sem="relaxed")
        tl.atomic_add(out_sum_ptr + channels, sum_where, sem="relaxed")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 19:
        raise ValueError(f"expected 19 inputs, got {len(inputs)}")

    (
        where_12,
        getitem_48,
        arg55_1,
        full,
        arg52_1,
        arg7_1,
        arg54_1,
        arg53_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
        shape7,
        shape8,
        shape9,
        shape10,
    ) = inputs

    tensors = (
        where_12,
        getitem_48,
        arg55_1,
        full,
        arg52_1,
        arg7_1,
        arg54_1,
        arg53_1,
    )
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("first eight repro inputs must be tensors")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("all tensor inputs must be torch.float32")
    if any(not x.is_cuda for x in tensors):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")

    expected_4d = (N, C, H, W)
    if tuple(where_12.shape) != expected_4d:
        raise ValueError(f"unexpected where_12 shape: {tuple(where_12.shape)}")
    if tuple(getitem_48.shape) != expected_4d:
        raise ValueError(f"unexpected getitem_48 shape: {tuple(getitem_48.shape)}")
    if tuple(arg55_1.shape) != expected_4d:
        raise ValueError(f"unexpected arg55_1 shape: {tuple(arg55_1.shape)}")
    if tuple(arg52_1.shape) != expected_4d:
        raise ValueError(f"unexpected arg52_1 shape: {tuple(arg52_1.shape)}")
    if tuple(full.shape) != ():
        raise ValueError(f"unexpected full shape: {tuple(full.shape)}")
    if tuple(arg7_1.shape) != (C,):
        raise ValueError(f"unexpected arg7_1 shape: {tuple(arg7_1.shape)}")
    if tuple(arg54_1.shape) != (N, GROUPS):
        raise ValueError(f"unexpected arg54_1 shape: {tuple(arg54_1.shape)}")
    if tuple(arg53_1.shape) != (N, GROUPS):
        raise ValueError(f"unexpected arg53_1 shape: {tuple(arg53_1.shape)}")

    if not all(
        x.is_contiguous()
        for x in (
            where_12,
            getitem_48,
            arg55_1,
            arg52_1,
            arg7_1,
            arg54_1,
            arg53_1,
        )
    ):
        raise ValueError("all non-scalar tensor inputs must be contiguous")

    expected_shapes = (
        [N, C, HW],
        [N, C, HW],
        [N, GROUPS, CHANNELS_PER_GROUP],
        [N, GROUPS, CHANNELS_PER_GROUP],
        [1, GROUPS, CHANNELS_PER_GROUP],
        [N, GROUPS, CHANNELS_PER_GROUP, HW],
        [N, GROUPS, CHANNELS_PER_GROUP, HW],
        [N, C, H, W],
        [N, GROUPS, CHANNELS_PER_GROUP],
        [N, GROUPS, CHANNELS_PER_GROUP],
        [C],
    )
    actual_shapes = (
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
        shape7,
        shape8,
        shape9,
        shape10,
    )
    for idx, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes, strict=True)):
        if list(actual) != expected:
            raise ValueError(f"unexpected shape param {idx}: {actual!r} != {expected!r}")

    return tensors


@oracle_impl(hardware="H100", shapes="(T([64, 64, 8, 8], f32), T([64, 64, 8, 8], f32), T([64, 64, 8, 8], f32), T([], f32), T([64, 64, 8, 8], f32), T([64], f32), T([64, 32], f32), T([64, 32], f32), S([64, 64, 64]), S([64, 64, 64]), S([64, 32, 2]), S([64, 32, 2]), S([1, 32, 2]), S([64, 32, 2, 64]), S([64, 32, 2, 64]), S([64, 64, 8, 8]), S([64, 32, 2]), S([64, 32, 2]), S([64]))")
def oracle_forward(inputs):
    """Run the full-scope Triton oracle for Repro.forward()."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        where_12,
        getitem_48,
        arg55_1,
        full,
        arg52_1,
        arg7_1,
        arg54_1,
        arg53_1,
    ) = _validate_inputs(inputs)

    out_full = torch.empty_like(where_12)
    out_vec = torch.empty((C,), device=where_12.device, dtype=torch.float32)
    out_sum = torch.empty((C,), device=where_12.device, dtype=torch.float32)

    _zero_vectors_kernel[(1,)](
        out_vec,
        out_sum,
        N_COLS=C,
        BLOCK=triton.next_power_of_2(C),
        num_warps=2,
    )
    _grouped_spatial_kernel[(N * GROUPS,)](
        where_12,
        getitem_48,
        arg55_1,
        full,
        arg52_1,
        arg7_1,
        arg54_1,
        arg53_1,
        out_full,
        out_vec,
        out_sum,
        GROUPS_=GROUPS,
        C_=C,
        CHANNELS_PER_GROUP_=CHANNELS_PER_GROUP,
        HW_=HW,
        GROUP_SCALE_=GROUP_SCALE,
        BLOCK_C=CHANNELS_PER_GROUP,
        BLOCK_HW=HW,
        num_warps=2,
    )
    return out_full, out_vec, out_sum


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
