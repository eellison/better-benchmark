"""
Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full
two-head `[128, 2]` ignore-index cross-entropy by reading each head directly
from the original strided input, using online logsumexp scalar accumulators for
both heads, folding the gather, valid-count division, and final average into a
single scalar Triton kernel. Inductor currently sees the decomposed
view/split/clone/amax/sub/exp/sum/log/gather/mask/count/div graph and cannot
recognize that the materialized log-softmax rows are only consumed by one
gathered negative log-likelihood per head. The required Inductor change is a
NEW_PATTERN lowering for small multi-head log_softmax plus gather plus
ignore-index cross-entropy means that emits scalar online accumulators and the
loss/count epilogue directly.
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

N_ROWS = 128
N_HEADS = 2
IGNORE_INDEX = 128

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
    def _two_head_xent_kernel(
        logits_ptr,
        label0_ptr,
        label1_ptr,
        fill_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        ignore_index: tl.constexpr,
        block_n: tl.constexpr,
    ):
        offsets = tl.arange(0, block_n)
        mask = offsets < n_rows

        raw_label0 = tl.load(label0_ptr)
        raw_label1 = tl.load(label1_ptr)
        label0 = tl.minimum(tl.maximum(raw_label0, 0), ignore_index)
        label1 = tl.minimum(tl.maximum(raw_label1, 0), ignore_index)
        valid0 = label0 != ignore_index
        valid1 = label1 != ignore_index
        safe_label0 = tl.where(valid0, label0, 0)
        safe_label1 = tl.where(valid1, label1, 0)

        target0 = tl.load(logits_ptr + safe_label0 * 2, mask=valid0, other=0.0).to(tl.float32)
        target1 = tl.load(logits_ptr + safe_label1 * 2 + 1, mask=valid1, other=0.0).to(tl.float32)

        row_max0 = tl.full([], -float("inf"), tl.float32)
        row_max1 = tl.full([], -float("inf"), tl.float32)
        denom0 = tl.full([], 0.0, tl.float32)
        denom1 = tl.full([], 0.0, tl.float32)

        for block_start in tl.range(0, n_rows, block_n):
            rows = block_start + offsets
            block_mask = mask & (rows < n_rows)
            x0 = tl.load(logits_ptr + rows * 2, mask=block_mask, other=-float("inf")).to(tl.float32)
            x1 = tl.load(logits_ptr + rows * 2 + 1, mask=block_mask, other=-float("inf")).to(tl.float32)

            block_max0 = tl.max(x0, axis=0)
            block_max1 = tl.max(x1, axis=0)
            new_max0 = tl.maximum(row_max0, block_max0)
            new_max1 = tl.maximum(row_max1, block_max1)
            denom0 = denom0 * tl.exp(row_max0 - new_max0) + tl.sum(tl.exp(x0 - new_max0), axis=0)
            denom1 = denom1 * tl.exp(row_max1 - new_max1) + tl.sum(tl.exp(x1 - new_max1), axis=0)
            row_max0 = new_max0
            row_max1 = new_max1

        fill = tl.load(fill_ptr).to(tl.float32)
        loss0 = row_max0 + tl.log(denom0) - target0
        loss1 = row_max1 + tl.log(denom1) - target1
        loss0 = tl.where(valid0, loss0, fill)
        loss1 = tl.where(valid1, loss1, fill)

        count0 = tl.where(valid0, 1.0, 0.0)
        count1 = tl.where(valid1, 1.0, 0.0)
        out = (loss0 / count0 + loss1 / count1) * 0.5
        tl.store(out_ptr, out)


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _launch_oracle(
    logits: torch.Tensor,
    label0: torch.Tensor,
    label1: torch.Tensor,
    fill: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int,
) -> torch.Tensor:
    _require_triton_cuda()
    if logits.shape != (N_ROWS, N_HEADS) or logits.dtype != torch.float32:
        raise ValueError(f"expected f32 logits shape {(N_ROWS, N_HEADS)}, got {tuple(logits.shape)} {logits.dtype}")
    if label0.shape != (1,) or label1.shape != (1,) or label0.dtype != torch.int64 or label1.dtype != torch.int64:
        raise ValueError("expected two int64 label tensors with shape [1]")
    if fill.shape != () or fill.dtype != torch.float32:
        raise ValueError(f"expected scalar f32 fill, got {tuple(fill.shape)} {fill.dtype}")
    if out.shape != () or out.dtype != torch.float32:
        raise ValueError(f"expected scalar f32 output, got {tuple(out.shape)} {out.dtype}")

    _two_head_xent_kernel[(1,)](
        logits,
        label0,
        label1,
        fill,
        out,
        n_rows=N_ROWS,
        ignore_index=IGNORE_INDEX,
        block_n=block_n,
        num_warps=4,
    )
    return out


def oracle_forward(inputs, *, block_n: int = 128) -> torch.Tensor:
    """Run the full-scope oracle computation.

    SCOPE INVARIANT: this accepts the exact input tuple from make_inputs() and
    returns the same scalar tensor as Repro.forward().
    """
    logits, label0, label1, fill, shape_param = inputs
    if tuple(shape_param) != (1, N_ROWS, N_HEADS):
        raise ValueError(f"unexpected view shape parameter: {shape_param}")
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    return _launch_oracle(logits, label0, label1, fill, out, block_n=block_n)


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
        lambda values: oracle_forward(values, block_n=args.block_n),
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
        oracle_forward(inputs, block_n=args.block_n)
        torch.cuda.synchronize()
        oracle_us = _bench_cuda(
            lambda: oracle_forward(inputs, block_n=args.block_n),
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
    true_floor = oracle_us < best_required_compile_us
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_results["coordinate_descent_tuning"], 3),
        "combo_compile_us": round(compile_results["combo_looped_cd"], 3),
        "best_required_compile_us": round(best_required_compile_us, 3),
        "ratio": round(best_required_compile_us / oracle_us, 3),
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "classification": "NEW_PATTERN",
    }
    print(json.dumps(result))
    if not true_floor:
        print("WARNING: oracle is slower than a required compile baseline")
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
