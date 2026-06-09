"""
Canonical-local oracle scaffold for repro sum_18262b26934c.

Repro pattern:
    grad[N, C, OH, OW]
    offsets[N, C, OH, OW]  # low-memory maxpool offsets in [0, 8]
    mask[N, C, IH, IW]
    full[]

    scatter = zeros([N * C, IH * IW]).scatter_add_(
        dim=1,
        index=maxpool_offsets_to_flat_indices(offsets),
        src=grad.view(N * C, OH * OW),
    ).view(N, C, IH, IW)
    out = where(mask, full, scatter).sum(dim=(0, 2, 3))

The key simplification is that the final reduction is over N/H/W for each
channel. The full scatter tensor is unnecessary: every maxpool output gradient
contributes directly to out[c] iff its selected input position is not masked;
masked input positions contribute exactly `full` once each.

This file contains:
  * a pure PyTorch direct formulation used as the correctness reference;
  * a two-stage Triton direct-reduction oracle that never materializes scatter;
  * CLI/benchmark/CSV append plumbing for measured_oracle_floors.csv.

TODOs for future tightening:
  * compare against the captured prims._low_memory_max_pool_offsets_to_indices
    implementation for odd padding/dilation cases if this is generalized beyond
    the current [3, 3], stride [2, 2], padding [0, 0], dilation [1, 1] repro;
  * explore fusing the mask-full count and scatter contribution reductions into
    one persistent/wave-specialized kernel for a lower measured floor.
"""
from __future__ import annotations

import argparse
import sys
import csv
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
    from triton.testing import do_bench
except Exception:  # pragma: no cover - keeps --help/direct mode usable without Triton.
    triton = None
    tl = None
    do_bench = None



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

REPRO_ID = "sum_18262b26934c"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = "repros/canonical/sum_18262b26934c/repro.py"
ORACLE_PATH = "repros/canonical/sum_18262b26934c/oracle_maxpool_direct_reduce.py"


if triton is not None:

    @triton.jit
    def _scatter_contrib_partials_kernel(
        grad_ptr,
        offsets_ptr,
        mask_ptr,
        partials_ptr,
        n_batches: tl.constexpr,
        n_channels: tl.constexpr,
        out_h: tl.constexpr,
        out_w: tl.constexpr,
        in_h: tl.constexpr,
        in_w: tl.constexpr,
        block: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        out_hw: tl.constexpr = out_h * out_w
        total: tl.constexpr = n_batches * out_h * out_w
        offsets = block_id * block + tl.arange(0, block)
        valid = offsets < total

        batch = offsets // out_hw
        rem = offsets - batch * out_hw
        out_y = rem // out_w
        out_x = rem - out_y * out_w

        dense_index = ((batch * n_channels + channel) * out_h + out_y) * out_w + out_x
        lowmem_offset = tl.load(offsets_ptr + dense_index, mask=valid, other=0).to(tl.int64)
        kernel_y = lowmem_offset // 3
        kernel_x = lowmem_offset - kernel_y * 3
        in_y = out_y * 2 + kernel_y
        in_x = out_x * 2 + kernel_x

        mask_index = ((batch * n_channels + channel) * in_h + in_y) * in_w + in_x
        masked = tl.load(mask_ptr + mask_index, mask=valid, other=1)
        grad = tl.load(grad_ptr + dense_index, mask=valid, other=0.0).to(tl.float32)
        contrib = tl.where(masked, 0.0, grad)
        partial = tl.sum(tl.where(valid, contrib, 0.0), axis=0)

        n_blocks: tl.constexpr = tl.cdiv(total, block)
        tl.store(partials_ptr + channel * n_blocks + block_id, partial)


    @triton.jit
    def _mask_full_partials_kernel(
        mask_ptr,
        full_ptr,
        partials_ptr,
        n_batches: tl.constexpr,
        n_channels: tl.constexpr,
        in_h: tl.constexpr,
        in_w: tl.constexpr,
        block: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        in_hw: tl.constexpr = in_h * in_w
        total: tl.constexpr = n_batches * in_h * in_w
        offsets = block_id * block + tl.arange(0, block)
        valid = offsets < total

        batch = offsets // in_hw
        rem = offsets - batch * in_hw
        in_y = rem // in_w
        in_x = rem - in_y * in_w
        mask_index = ((batch * n_channels + channel) * in_h + in_y) * in_w + in_x
        masked = tl.load(mask_ptr + mask_index, mask=valid, other=0)
        count = tl.sum(tl.where(masked & valid, 1.0, 0.0), axis=0)
        full = tl.load(full_ptr).to(tl.float32)

        n_blocks: tl.constexpr = tl.cdiv(total, block)
        tl.store(partials_ptr + channel * n_blocks + block_id, count * full)


    @triton.jit
    def _finalize_channel_sums_kernel(
        scatter_partials_ptr,
        mask_partials_ptr,
        out_ptr,
        n_scatter_blocks: tl.constexpr,
        n_mask_blocks: tl.constexpr,
        reduce_block: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, reduce_block)
        scatter = tl.load(
            scatter_partials_ptr + channel * n_scatter_blocks + offsets,
            mask=offsets < n_scatter_blocks,
            other=0.0,
        ).to(tl.float32)
        mask_full = tl.load(
            mask_partials_ptr + channel * n_mask_blocks + offsets,
            mask=offsets < n_mask_blocks,
            other=0.0,
        ).to(tl.float32)
        tl.store(out_ptr + channel, tl.sum(scatter + mask_full, axis=0))


def maxpool_offsets_to_flat_indices(
    offsets: torch.Tensor,
    in_h: int,
    in_w: int,
    stride_h: int = 2,
    stride_w: int = 2,
) -> torch.Tensor:
    """Convert current-repro low-memory 3x3 maxpool offsets to IH*IW indices."""
    if offsets.ndim != 4:
        raise ValueError(f"expected offsets[N,C,OH,OW], got shape {tuple(offsets.shape)}")
    _, _, out_h, out_w = offsets.shape
    out_y = torch.arange(out_h, device=offsets.device).view(1, 1, out_h, 1)
    out_x = torch.arange(out_w, device=offsets.device).view(1, 1, 1, out_w)
    offsets_i64 = offsets.to(torch.int64)
    kernel_y = offsets_i64 // 3
    kernel_x = offsets_i64 - kernel_y * 3
    in_y = out_y * stride_h + kernel_y
    in_x = out_x * stride_w + kernel_x
    return (in_y * in_w + in_x).view(offsets.shape[0], offsets.shape[1], out_h * out_w)


def direct_pytorch_reference(
    grad: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
    full: torch.Tensor,
) -> torch.Tensor:
    """Direct channel reduction equivalent to the repro, without scatter materialization."""
    if grad.ndim != 4 or offsets.shape != grad.shape or mask.ndim != 4:
        raise ValueError("expected grad/offsets[N,C,OH,OW] and mask[N,C,IH,IW]")
    n_batches, n_channels, out_h, out_w = grad.shape
    if mask.shape[:2] != (n_batches, n_channels):
        raise ValueError(f"mask batch/channel shape {tuple(mask.shape[:2])} does not match grad")

    in_h, in_w = mask.shape[2:]
    flat_indices = maxpool_offsets_to_flat_indices(offsets, in_h, in_w)
    selected_mask = torch.gather(mask.view(n_batches, n_channels, in_h * in_w), 2, flat_indices)
    scatter_part = (grad.view(n_batches, n_channels, out_h * out_w) * (~selected_mask).to(grad.dtype)).sum(dim=(0, 2))
    full_part = mask.to(grad.dtype).sum(dim=(0, 2, 3)) * full.to(grad.dtype)
    return scatter_part + full_part


def repro_equivalent_reference(
    grad: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
    full: torch.Tensor,
) -> torch.Tensor:
    """Literal scatter/where/sum reference; use only for small shapes."""
    n_batches, n_channels, out_h, out_w = grad.shape
    in_h, in_w = mask.shape[2:]
    flat_indices = maxpool_offsets_to_flat_indices(offsets, in_h, in_w)
    scatter = torch.zeros((n_batches * n_channels, in_h * in_w), device=grad.device, dtype=grad.dtype)
    scatter.scatter_add_(1, flat_indices.view(n_batches * n_channels, out_h * out_w), grad.view(n_batches * n_channels, out_h * out_w))
    return torch.where(mask, full, scatter.view(n_batches, n_channels, in_h, in_w)).sum(dim=(0, 2, 3))


def triton_direct_oracle(
    grad: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
    full: torch.Tensor,
    block: int = 1024,
    warps: int = 4,
) -> torch.Tensor:
    """Two-stage Triton oracle for the current maxpool-offset direct reduction."""
    if triton is None:
        raise RuntimeError("Triton is not available; use --impl pytorch for the direct reference")
    if not (grad.is_cuda and offsets.is_cuda and mask.is_cuda and full.is_cuda):
        raise ValueError("Triton oracle requires CUDA tensors")
    if grad.dtype != torch.float32 or offsets.dtype != torch.int8 or mask.dtype != torch.bool:
        raise ValueError("expected grad=float32, offsets=int8, mask=bool")

    n_batches, n_channels, out_h, out_w = grad.shape
    in_h, in_w = mask.shape[2:]
    scatter_total = n_batches * out_h * out_w
    mask_total = n_batches * in_h * in_w
    scatter_blocks = triton.cdiv(scatter_total, block)
    mask_blocks = triton.cdiv(mask_total, block)
    scatter_partials = torch.empty((n_channels, scatter_blocks), device=grad.device, dtype=torch.float32)
    mask_partials = torch.empty((n_channels, mask_blocks), device=grad.device, dtype=torch.float32)
    out = torch.empty((n_channels,), device=grad.device, dtype=torch.float32)

    _scatter_contrib_partials_kernel[(n_channels, scatter_blocks)](
        grad,
        offsets,
        mask,
        scatter_partials,
        n_batches,
        n_channels,
        out_h,
        out_w,
        in_h,
        in_w,
        block=block,
        num_warps=warps,
    )
    _mask_full_partials_kernel[(n_channels, mask_blocks)](
        mask,
        full,
        mask_partials,
        n_batches,
        n_channels,
        in_h,
        in_w,
        block=block,
        num_warps=warps,
    )
    reduce_block = triton.next_power_of_2(max(scatter_blocks, mask_blocks))
    _finalize_channel_sums_kernel[(n_channels,)](
        scatter_partials,
        mask_partials,
        out,
        scatter_blocks,
        mask_blocks,
        reduce_block=reduce_block,
        num_warps=8,
    )
    return out


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    grad = torch.randn((args.n, args.c, args.oh, args.ow), device=device, dtype=torch.float32, generator=generator)
    offsets = torch.randint(0, 9, (args.n, args.c, args.oh, args.ow), device=device, dtype=torch.int8, generator=generator)
    mask = torch.rand((args.n, args.c, args.ih, args.iw), device=device, dtype=torch.float32, generator=generator) < args.mask_prob
    full = torch.randn((), device=device, dtype=torch.float32, generator=generator)
    return grad, offsets, mask, full


def benchmark_us(fn: Callable[[], torch.Tensor], warmup: int, rep: int) -> float:
    if do_bench is not None:
        return float(do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best_ms = math.inf
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best_ms = min(best_ms, float(start.elapsed_time(end)))
    return best_ms * 1000.0


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_baseline_row() -> dict[str, str]:
    path = Path("investigation_results/sol_gap_candidates.csv")
    if not path.exists():
        return {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("repro_id") == REPRO_ID:
                return row
    return {}


def append_measurement(args: argparse.Namespace, oracle_us: float, correct: str, max_abs_diff: float) -> None:
    baseline = load_baseline_row()
    best_compile_us = float(baseline.get("best_compile_us", "nan"))
    memcopy_sol_us = float(baseline.get("memcopy_sol_us", "nan"))
    total_bytes = int(float(baseline.get("total_bytes", "0") or 0))
    n_kernels = int(float(baseline.get("n_kernels", "0") or 0))

    row = {
        "repro_id": REPRO_ID,
        "repro_path": REPRO_PATH,
        "shape_label": f"N{args.n}_C{args.c}_OH{args.oh}_OW{args.ow}_IH{args.ih}_IW{args.iw}",
        "family": "maxpool_backward_masked_channel_reduce",
        "oracle_impl": f"{args.impl}_maxpool_offset_direct_channel_reduce",
        "oracle_path": ORACLE_PATH,
        "hardware": args.hardware,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no_cuda",
        "git_commit": get_git_commit(),
        "compiled_us": baseline.get("compiled_us", ""),
        "coord_descent_us": baseline.get("coord_descent_us", ""),
        "best_compile_us": best_compile_us,
        "memcopy_sol_us": memcopy_sol_us,
        "oracle_us": oracle_us,
        "total_bytes": total_bytes,
        "n_kernels": n_kernels,
        "oracle_over_sol": oracle_us / memcopy_sol_us if memcopy_sol_us == memcopy_sol_us else math.nan,
        "speedup_vs_best_compile": best_compile_us / oracle_us if best_compile_us == best_compile_us else math.nan,
        "correct": correct,
        "max_abs_diff": max_abs_diff,
        "tolerance": "rtol=1e-4,atol=1e-3" if correct != "not_checked" else "not_checked",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "Direct channel reduction for maxpool-offset scatter_add+where+sum; avoids materializing [N*C,IH*IW] scatter.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    with args.out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended {args.out}")


@oracle_impl(hardware="H100", shapes="(T([512, 64, 55, 55], f32), T([512, 64, 55, 55], i8, gen=Index(9)), T([512, 64, 111, 111], b8), T([], f32), S([32768, 3025]), S([32768, 3025]), S([512, 64, 111, 111]))")
def oracle_forward(inputs):
    return triton_direct_oracle(*inputs)


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
