"""
Full-scope Triton oracle for sum_ab7572b27d2c (LayoutLM attention layout +
feature sum).

Gap diagnosis (classification: SCHEDULER_FUSION): This oracle covers the full
compiled repro.py scope, not just the reduction subset: it consumes the original
strided `[32, 12, 512, 64]` input, writes the contiguous clone backing storage
for the returned `[768, 16384]` strided view, and computes the `[768]` feature
sum from the same input pass. Inductor currently emits this as a layout clone
plus a separate reduction because the scheduler does not fuse a materializing
layout-copy output with a reduction output that share the same source iteration
space but expose different result layouts. This is a SCHEDULER_FUSION gap if
the full-scope Triton timing beats both required compile configurations; if it
does not, keep this artifact as diagnosis-only rather than a true floor.
"""
from __future__ import annotations

import argparse
import importlib.util
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



REPRO_ID = "sum_ab7572b27d2c"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 12
SEQ = 512
HEAD_DIM = 64
ROWS = BATCH * SEQ
FEATURES = HEADS * HEAD_DIM



@triton.jit
def _copy_reduce_partials_kernel(
    x_ptr,
    clone_ptr,
    partials_ptr,
    x_stride_b: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_s: tl.constexpr,
    x_stride_d: tl.constexpr,
    SEQ_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    ROWS_: tl.constexpr,
    FEATURES_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    feature_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    batch = row_offsets // SEQ_
    seq = row_offsets - batch * SEQ_
    head = feature_offsets // HEAD_DIM_
    dim = feature_offsets - head * HEAD_DIM_

    x_offsets = (
        batch[:, None] * x_stride_b
        + head[None, :] * x_stride_h
        + seq[:, None] * x_stride_s
        + dim[None, :] * x_stride_d
    )
    clone_offsets = row_offsets[:, None] * FEATURES_ + feature_offsets[None, :]
    mask = (row_offsets[:, None] < ROWS_) & (feature_offsets[None, :] < FEATURES_)

    values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(clone_ptr + clone_offsets, values, mask=mask)

    partial = tl.sum(values, axis=0)
    tl.store(
        partials_ptr + pid_m * FEATURES_ + feature_offsets,
        partial,
        mask=feature_offsets < FEATURES_,
    )


@triton.jit
def _finish_partials_kernel(
    partials_ptr,
    sum_ptr,
    num_row_blocks: tl.constexpr,
    FEATURES_: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    feature = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_R)
    mask = row_blocks < num_row_blocks
    values = tl.load(
        partials_ptr + row_blocks * FEATURES_ + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(sum_ptr + feature, tl.sum(values, axis=0))


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    getitem_45: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert getitem_45.shape == (BATCH, HEADS, SEQ, HEAD_DIM)
    assert getitem_45.dtype is torch.float32

    clone_storage = torch.empty((ROWS, FEATURES), device=getitem_45.device, dtype=torch.float32)
    sums = torch.empty((FEATURES,), device=getitem_45.device, dtype=torch.float32)

    block_m = 64
    block_n = 32
    num_row_blocks = triton.cdiv(ROWS, block_m)
    partials = torch.empty((num_row_blocks, FEATURES), device=getitem_45.device, dtype=torch.float32)

    _copy_reduce_partials_kernel[(num_row_blocks, triton.cdiv(FEATURES, block_n))](
        getitem_45,
        clone_storage,
        partials,
        x_stride_b=getitem_45.stride(0),
        x_stride_h=getitem_45.stride(1),
        x_stride_s=getitem_45.stride(2),
        x_stride_d=getitem_45.stride(3),
        SEQ_=SEQ,
        HEAD_DIM_=HEAD_DIM,
        ROWS_=ROWS,
        FEATURES_=FEATURES,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    _finish_partials_kernel[(FEATURES,)](
        partials,
        sums,
        num_row_blocks=num_row_blocks,
        FEATURES_=FEATURES,
        BLOCK_R=triton.next_power_of_2(num_row_blocks),
        num_warps=8,
    )

    layout_view = torch.as_strided(clone_storage, (FEATURES, ROWS), (1, FEATURES))
    return layout_view, sums


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


@oracle_impl(hardware="H100", shapes="(T([32, 12, 512, 64], f32, stride=(393216, 64, 768, 1)), S([32, 512, 768]), S([16384, 768]), S([768]))")
def oracle_forward(inputs):
    return oracle_fused(*inputs)


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
