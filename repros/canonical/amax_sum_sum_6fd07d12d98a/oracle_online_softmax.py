"""
Oracle for amax_sum_sum_6fd07d12d98a

Gap diagnosis:
  Classification: NEW_PATTERN
  What oracle does differently: Computes the full shifted-label ignore-index cross-entropy mean with Triton online logsumexp scalar accumulators and a scalar mean reduction, without materializing the row log-softmax.
  What Inductor change would fix: Recognize shifted-label log_softmax plus gather plus masked mean cross entropy and lower it to an online row-accumulator template with a small reduction epilogue.
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

HISTORICAL_BEST_COMPILE_US = 1098.7520217895508

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
    def _shifted_online_xent_rows_kernel(
        labels_ptr,
        logits_ptr,
        loss_ptr,
        valid_ptr,
        seq_len: tl.constexpr,
        n_cols: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // seq_len
        pos = row - batch * seq_len
        has_next = pos + 1 < seq_len

        label = tl.load(
            labels_ptr + batch * seq_len + pos + 1,
            mask=has_next,
            other=-100,
        )
        is_valid = label != -100
        safe_label = tl.where(is_valid, label, 0)

        row_start = row * n_cols
        target = tl.load(
            logits_ptr + row_start + safe_label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)

        row_max = tl.full([], -float("inf"), tl.float32)
        denom = tl.full([], 0.0, tl.float32)

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + tl.arange(0, block_n)
            mask = cols < n_cols
            x = tl.load(
                logits_ptr + row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)

            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(row_max, block_max)
            denom = denom * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
            row_max = new_max

        loss = row_max + tl.log(denom) - target
        loss = tl.where(is_valid, loss, 0.0)

        tl.store(loss_ptr + row, loss)
        tl.store(valid_ptr + row, tl.where(is_valid, 1.0, 0.0))

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
    labels: torch.Tensor,
    logits: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    _require_triton_cuda()

    if labels.ndim != 2 or labels.dtype != torch.int64:
        raise ValueError(f"expected i64 labels[batch, seq], got shape={tuple(labels.shape)} dtype={labels.dtype}")
    if logits.ndim != 3 or logits.dtype != torch.float32:
        raise ValueError(f"expected f32 logits[batch, seq, vocab], got shape={tuple(logits.shape)} dtype={logits.dtype}")
    if labels.shape[0] != logits.shape[0] or labels.shape[1] != logits.shape[1]:
        raise ValueError(f"labels shape {tuple(labels.shape)} does not match logits prefix {tuple(logits.shape[:2])}")

    batch, seq_len = labels.shape
    n_cols = logits.shape[2]
    n_rows = batch * seq_len
    if loss_per_row.shape != (n_rows,) or valid_per_row.shape != (n_rows,):
        raise ValueError("temporary buffers do not match flattened row count")
    if out.shape != () or out.dtype != torch.float32:
        raise ValueError(f"expected scalar f32 output, got shape={tuple(out.shape)} dtype={out.dtype}")

    _shifted_online_xent_rows_kernel[(n_rows,)](
        labels,
        logits,
        loss_per_row,
        valid_per_row,
        seq_len=seq_len,
        n_cols=n_cols,
        block_n=block_n,
        num_warps=num_warps,
    )
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        out,
        n_rows=n_rows,
        block_m=triton.next_power_of_2(n_rows),
        num_warps=8,
    )
    return out


def oracle_forward(inputs, *, block_n: int = 4096, num_warps: int = 8):
    """Run the full-scope oracle computation.

    SCOPE INVARIANT: accepts the exact input tuple from make_inputs() and
    returns the same scalar f32 mean loss as Repro.forward().
    """
    labels, logits, _shape_param_0 = inputs
    del _shape_param_0

    n_rows = labels.numel()
    loss_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    return _launch_oracle(
        labels,
        logits,
        loss_per_row,
        valid_per_row,
        out,
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
    if not args.no_compile:
        with torch.no_grad():
            for label, config in COMPILE_CONFIGS:
                compiled = _compile_with_config(instance, inputs, config, warmup=args.warmup)
                us = _bench_cuda(lambda: compiled(*inputs), warmup=args.warmup, rep=args.rep)
                compile_results[label] = us
                del compiled
                torch.cuda.empty_cache()

    best_required_compile_us = min(compile_results.values()) if compile_results else None
    best_compare_us = (
        min(best_required_compile_us, HISTORICAL_BEST_COMPILE_US)
        if best_required_compile_us is not None
        else HISTORICAL_BEST_COMPILE_US
    )
    true_floor = oracle_us < best_compare_us
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": (
            round(compile_results["coordinate_descent_tuning"], 3)
            if "coordinate_descent_tuning" in compile_results
            else None
        ),
        "combo_compile_us": (
            round(compile_results["combo_looped_cd"], 3)
            if "combo_looped_cd" in compile_results
            else None
        ),
        "best_required_compile_us": (
            round(best_required_compile_us, 3)
            if best_required_compile_us is not None
            else None
        ),
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "ratio": (
            round(best_required_compile_us / oracle_us, 3)
            if best_required_compile_us is not None
            else None
        ),
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
