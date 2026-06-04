"""
Oracle for amax_sum_sum_c9a6620c3408

Gap diagnosis:
  Classification: NEW_PATTERN
  What oracle does differently: Computes the full MobileBERT masked-LM forward slice, bias add, ignore-index cross-entropy mean, and materialized softmax output with one online row kernel plus one scalar reduction.
  What Inductor change would fix: Add a semantic lowering for biased log_softmax plus gather plus masked mean that can share online softmax row accumulators with required sibling materialized outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from oracle_harness import (
    check_oracle,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

HISTORICAL_BEST_COMPILE_US = 2980.9279441833496

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _full_online_softmax_xent_kernel(
        mm_ptr,
        bias_ptr,
        labels_ptr,
        fill_ptr,
        logits_out_ptr,
        probs_out_ptr,
        loss_ptr,
        valid_ptr,
        n_cols: tl.constexpr,
        mm_row_stride: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols_base = tl.arange(0, block_n)
        mm_row_start = row * mm_row_stride
        out_row_start = row * n_cols

        label = tl.load(labels_ptr + row)
        is_valid = label != -100
        safe_label = tl.where(is_valid, label, 0)
        fill = tl.load(fill_ptr).to(tl.float32)
        target = tl.load(
            mm_ptr + mm_row_start + safe_label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)
        target += tl.load(bias_ptr + safe_label, mask=is_valid, other=0.0).to(tl.float32)

        row_max = tl.full([], -float("inf"), tl.float32)
        denom = tl.full([], 0.0, tl.float32)

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + cols_base
            mask = cols < n_cols
            x = tl.load(
                mm_ptr + mm_row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + b
            tl.store(logits_out_ptr + out_row_start + cols, x, mask=mask)

            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(row_max, block_max)
            denom = denom * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
            row_max = new_max

        loss = row_max + tl.log(denom) - target
        loss = tl.where(is_valid, loss, fill)
        tl.store(loss_ptr + row, loss)
        tl.store(valid_ptr + row, tl.where(is_valid, 1.0, 0.0))

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + cols_base
            mask = cols < n_cols
            x = tl.load(
                logits_out_ptr + out_row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            probs = tl.exp(x - row_max) / denom
            tl.store(probs_out_ptr + out_row_start + cols, probs, mask=mask)

    @triton.jit
    def _mean_reduce_kernel(
        loss_ptr,
        valid_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        block_m: tl.constexpr,
    ):
        offsets = tl.arange(0, block_m)
        mask = offsets < n_rows
        losses = tl.load(loss_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        valid = tl.load(valid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        total_loss = tl.sum(losses, axis=0)
        total_valid = tl.sum(valid, axis=0)
        tl.store(out_ptr, total_loss / total_valid)


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _launch_oracle(
    mm: torch.Tensor,
    bias: torch.Tensor,
    labels: torch.Tensor,
    fill: torch.Tensor,
    logits_out: torch.Tensor,
    probs_out: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    mean_out: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_cuda()
    n_rows = labels.numel()
    n_cols = bias.numel()

    if mm.ndim != 2 or mm.dtype != torch.float32:
        raise ValueError(f"expected f32 2D mm, got shape={tuple(mm.shape)} dtype={mm.dtype}")
    if bias.ndim != 1 or bias.dtype != torch.float32:
        raise ValueError(f"expected f32 1D bias, got shape={tuple(bias.shape)} dtype={bias.dtype}")
    if labels.dtype != torch.int64:
        raise ValueError(f"expected int64 labels, got dtype={labels.dtype}")
    if fill.shape != () or fill.dtype != torch.float32:
        raise ValueError(f"expected scalar f32 fill, got shape={tuple(fill.shape)} dtype={fill.dtype}")
    if mm.shape[0] != n_rows or mm.shape[1] < n_cols:
        raise ValueError(f"mm shape {tuple(mm.shape)} does not cover labels={n_rows}, bias={n_cols}")
    if logits_out.numel() != n_rows * n_cols or probs_out.shape != (n_rows, n_cols):
        raise ValueError("output shapes do not match the logical repro outputs")

    _full_online_softmax_xent_kernel[(n_rows,)](
        mm,
        bias,
        labels,
        fill,
        logits_out,
        probs_out,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        mm_row_stride=mm.stride(0),
        block_n=block_n,
        num_warps=num_warps,
    )
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        mean_out,
        n_rows=n_rows,
        block_m=triton.next_power_of_2(n_rows),
        num_warps=8,
    )
    return logits_out, mean_out, probs_out


def oracle_forward(inputs, *, block_n: int = 2048, num_warps: int = 8):
    """Run the full-scope oracle computation.

    SCOPE INVARIANT: this accepts the exact input tuple from make_inputs() and
    returns the same three tensors as Repro.forward().
    """
    mm, bias, labels, fill, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    n_rows = labels.numel()
    n_cols = bias.numel()
    logits_out = torch.empty(tuple(_shape_param_2), device=mm.device, dtype=torch.float32)
    probs_out = torch.empty((n_rows, n_cols), device=mm.device, dtype=torch.float32)
    loss_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    mean_out = torch.empty((), device=mm.device, dtype=torch.float32)

    labels_1d = labels.reshape(-1)
    return _launch_oracle(
        mm,
        bias,
        labels_1d,
        fill,
        logits_out,
        probs_out,
        loss_per_row,
        valid_per_row,
        mean_out,
        block_n=block_n,
        num_warps=num_warps,
    )


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    from triton.testing import do_bench

    return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _compile_with_config(instance, inputs, config: dict[str, object], warmup: int):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = instance.__class__().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_check(args: argparse.Namespace) -> bool:
    inputs = get_inputs()
    instance = get_repro_instance()
    ok = check_oracle(
        lambda values: oracle_forward(values, block_n=args.block_n, num_warps=args.num_warps),
        instance,
        inputs,
        atol=args.atol,
        rtol=args.rtol,
        skip_stochastic=False,
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(args: argparse.Namespace) -> dict[str, object]:
    inputs = get_inputs()
    instance = get_repro_instance()

    with torch.no_grad():
        oracle_forward(inputs, block_n=args.block_n, num_warps=args.num_warps)
        torch.cuda.synchronize()
        oracle_us = _bench_cuda(
            lambda: oracle_forward(inputs, block_n=args.block_n, num_warps=args.num_warps),
            warmup=args.warmup,
            rep=args.rep,
        )

    compile_results: dict[str, float] = {}
    with torch.no_grad():
        for label, config in COMPILE_CONFIGS:
            compiled = _compile_with_config(instance, inputs, config, warmup=args.warmup)
            us = _bench_cuda(lambda: compiled(*inputs), warmup=args.warmup, rep=args.rep)
            compile_results[label] = us
            del compiled
            torch.cuda.empty_cache()

    best_required_compile_us = min(compile_results.values())
    best_compare_us = min(best_required_compile_us, HISTORICAL_BEST_COMPILE_US)
    true_floor = oracle_us < best_compare_us
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_results["coordinate_descent_tuning"], 3),
        "combo_compile_us": round(compile_results["combo_looped_cd"], 3),
        "best_required_compile_us": round(best_required_compile_us, 3),
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "ratio": round(best_required_compile_us / oracle_us, 3),
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "classification": "NEW_PATTERN",
    }
    print(json.dumps(result))
    if not true_floor:
        print("WARNING: oracle is slower than a required local or historical compile baseline")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle and required compile configs")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument("--block-n", type=int, default=2048, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=8, help="Triton warps for the row kernel")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = args.bench = True

    _require_triton_cuda()
    with torch.no_grad():
        if args.check and not run_check(args):
            sys.exit(1)
        if args.bench:
            run_bench(args)


if __name__ == "__main__":
    main()
