"""
Full-scope Triton oracle for sum_2e0e9617102b.

The compiled repro computes:

    out[c] = sum_{n,h,w}(permute_46[n,c,h,w] + getitem_123[n,c,h,w])

for two original contiguous f32[128, 80, 56, 56] inputs and returns one
contiguous f32[80] tensor. This oracle covers that full add+channel-reduction
scope; it does not time a reduction-only subset.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle differs from the
eager expression by never materializing the full add tensor, instead reading the
two original inputs once and accumulating per-channel partials in Triton. That
is also the optimization Inductor can already express for this simple add+sum
region, so any remaining gap is expected to be a memory-bandwidth/tuning issue
rather than missing scheduler fusion, scatter-reduce handling, split-K,
algebraic elimination, or recomputation fusion. If this full-scope Triton
timing does not beat both required compile configurations, this artifact is
diagnosis-only rather than a true floor.
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



REPRO_ID = "sum_2e0e9617102b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 80
H = 56
W = 56
HW = H * W
BLOCK_HW = triton.next_power_of_2(HW)
DEFAULT_GROUP_N = 2



@triton.jit
def _add_sum_partials_kernel(
    x_ptr,
    y_ptr,
    partials_ptr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    y_stride_n: tl.constexpr,
    y_stride_c: tl.constexpr,
    y_stride_h: tl.constexpr,
    y_stride_w: tl.constexpr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_HW_: tl.constexpr,
):
    c = tl.program_id(0)
    group = tl.program_id(1)
    hw = tl.arange(0, BLOCK_HW_)
    h = hw // W_
    w = hw - h * W_
    hw_mask = hw < HW_

    acc = tl.zeros([BLOCK_HW_], dtype=tl.float32)
    for i in range(GROUP_N):
        n = group * GROUP_N + i
        valid = (n < N_) & hw_mask
        x_offsets = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
        y_offsets = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
        x_vals = tl.load(x_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)
        y_vals = tl.load(y_ptr + y_offsets, mask=valid, other=0.0).to(tl.float32)
        acc += x_vals + y_vals

    partial = tl.sum(acc, axis=0)
    tl.store(partials_ptr + group * C_ + c, partial, mask=group < NUM_GROUPS)


@triton.jit
def _finish_partials_kernel(
    partials_ptr,
    out_ptr,
    C_: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    c = tl.program_id(0)
    group_offsets = tl.arange(0, BLOCK_GROUPS)
    mask = group_offsets < NUM_GROUPS
    values = tl.load(partials_ptr + group_offsets * C_ + c, mask=mask, other=0.0)
    tl.store(out_ptr + c, tl.sum(values, axis=0))


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def _num_groups(group_n: int) -> int:
    return triton.cdiv(N, group_n)


def _make_workspace(device: torch.device, group_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    partials = torch.empty((_num_groups(group_n), C), device=device, dtype=torch.float32)
    out = torch.empty((C,), device=device, dtype=torch.float32)
    return partials, out


def _oracle_into(
    permute_46: torch.Tensor,
    getitem_123: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    group_n: int,
) -> torch.Tensor:
    assert permute_46.shape == (N, C, H, W)
    assert getitem_123.shape == (N, C, H, W)
    assert permute_46.dtype is torch.float32
    assert getitem_123.dtype is torch.float32
    assert out.shape == (C,)
    assert out.dtype is torch.float32

    num_groups = _num_groups(group_n)
    _add_sum_partials_kernel[(C, num_groups)](
        permute_46,
        getitem_123,
        partials,
        x_stride_n=permute_46.stride(0),
        x_stride_c=permute_46.stride(1),
        x_stride_h=permute_46.stride(2),
        x_stride_w=permute_46.stride(3),
        y_stride_n=getitem_123.stride(0),
        y_stride_c=getitem_123.stride(1),
        y_stride_h=getitem_123.stride(2),
        y_stride_w=getitem_123.stride(3),
        N_=N,
        C_=C,
        HW_=HW,
        W_=W,
        NUM_GROUPS=num_groups,
        GROUP_N=group_n,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
    )
    _finish_partials_kernel[(C,)](
        partials,
        out,
        C_=C,
        NUM_GROUPS=num_groups,
        BLOCK_GROUPS=triton.next_power_of_2(num_groups),
        num_warps=1,
    )
    return out


def oracle_fused(
    permute_46: torch.Tensor,
    getitem_123: torch.Tensor,
    group_n: int = DEFAULT_GROUP_N,
) -> torch.Tensor:
    partials, out = _make_workspace(permute_46.device, group_n)
    return _oracle_into(permute_46, getitem_123, partials, out, group_n)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        out = model(*inputs)
    return (out,) if isinstance(out, torch.Tensor) else tuple(out)


@oracle_impl(hardware="H100", shapes="(T([128, 80, 56, 56], f32), T([128, 80, 56, 56], f32))")
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
