"""
Canonical-local oracle scaffold for repro sum_sum_sum_afd118ca907d.

Repro pattern:
    T5 attention-backward-like fragment returning two softmax-gradient tensors
    and two accumulated relative-position-bias gradients.  Each half computes
    the stable softmax backward identity

        dscore = softmax * (dprob - sum(dprob * softmax, dim=-1))

    then adds five residual gradient tensors, reduces across batch, permutes
    to ``[query, key, head]``, masks ``bucket == -1`` entries with ``full_1``,
    and accumulates into a ``[32, 8]`` bucket table via ``index_put(...,
    accumulate=True)``.

The runnable implementation is a direct PyTorch subgraph/reference that keeps
the two softmax-backward/attention-backward halves explicit.  It is meant to be
used as a correctness and benchmarking harness while a real fused Triton oracle
is developed in this canonical repro directory.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_afd118ca907d"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

BATCH = 8
HEADS = 8
N_BUCKETS = 32
SCALE = 1.1111111111111112
NEG_INF_F32 = -3.4028234663852886e38


def make_shape_params(seq_len: int) -> tuple[list[int], ...]:
    attention = [BATCH, HEADS, seq_len, seq_len]
    bmm = [BATCH * HEADS, seq_len, seq_len]
    expanded_mask = [BATCH, -1, seq_len, seq_len]
    return (attention, bmm, attention, bmm, attention, expanded_mask, attention, bmm, attention, bmm)


def make_inputs(device: torch.device, seq_len: int, seed: int) -> tuple[object, ...]:
    """Create repro-compatible synthetic inputs, scaled down by ``--seq-len``.

    The captured default is ``seq_len=1024`` and is very large.  Smaller values
    exercise the same graph structure for local validation and syntax-safe CLI
    use; use ``--seq-len 1024`` when measuring an actual oracle floor.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def randn(shape: tuple[int, ...] | list[int]) -> torch.Tensor:
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)

    def rand_bool(shape: tuple[int, ...] | list[int], p_true: float = 0.9) -> torch.Tensor:
        return (torch.rand(tuple(shape), generator=generator) < p_true).to(device)

    attention_shape = (BATCH, HEADS, seq_len, seq_len)
    bmm_shape = (BATCH * HEADS, seq_len, seq_len)
    bias_shape = (seq_len, seq_len, HEADS)
    sum_shape = (BATCH, HEADS, seq_len, 1)

    bucket_0 = torch.randint(-1, N_BUCKETS, (seq_len, seq_len), generator=generator, dtype=torch.int64).to(device)
    bucket_1 = torch.randint(-1, N_BUCKETS, (seq_len, seq_len), generator=generator, dtype=torch.int64).to(device)
    # Keep a few sentinel entries to exercise the ``bucket == -1`` path.
    if seq_len > 0:
        bucket_0[0, 0] = -1
        bucket_1[-1, -1] = -1

    return (
        randn(attention_shape),
        randn(attention_shape),
        randn(attention_shape),
        randn(attention_shape),
        randn(attention_shape),
        randn(bmm_shape),
        rand_bool(attention_shape),
        torch.softmax(randn(attention_shape), dim=-1),
        bucket_0,
        torch.tensor(0.0, dtype=torch.float32, device=device),
        randn(attention_shape),
        randn(attention_shape),
        randn(attention_shape),
        randn(attention_shape),
        randn(attention_shape),
        randn(bmm_shape),
        rand_bool(attention_shape),
        rand_bool((1, 1, seq_len, 1)),
        randn(bmm_shape),
        randn(bias_shape),
        randn(sum_shape),
        torch.rand(sum_shape, generator=generator, dtype=torch.float32).to(device) + 0.25,
        bucket_1,
        *make_shape_params(seq_len),
    )


def softmax_backward_from_probs(
    grad_probs: torch.Tensor,
    softmax_probs: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    dprob = grad_probs * keep_mask.to(torch.float32) * SCALE
    product = dprob * softmax_probs
    row_dot = product.sum(dim=-1, keepdim=True)
    return product - softmax_probs * row_dot


def relative_position_bias_grad(
    attention_grad: torch.Tensor,
    bucket: torch.Tensor,
    fill_value: torch.Tensor,
) -> torch.Tensor:
    per_position = attention_grad.sum(dim=0).permute(1, 2, 0).contiguous()
    masked = torch.where((bucket == -1).unsqueeze(-1), fill_value, per_position)
    out = torch.zeros((N_BUCKETS, HEADS), dtype=attention_grad.dtype, device=attention_grad.device)
    return out.index_put((bucket,), masked, accumulate=True)


def first_attention_backward(
    residuals: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    bmm_grad: torch.Tensor,
    keep_mask: torch.Tensor,
    softmax_probs: torch.Tensor,
    bucket: torch.Tensor,
    fill_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dscore = softmax_backward_from_probs(bmm_grad.view_as(softmax_probs), softmax_probs, keep_mask)
    attention_grad = residuals[0] + residuals[1] + residuals[2] + residuals[3] + residuals[4] + dscore
    bucket_grad = relative_position_bias_grad(attention_grad, bucket, fill_value)
    return dscore.reshape_as(bmm_grad), bucket_grad


def second_attention_backward(
    residuals: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    bmm_grad: torch.Tensor,
    keep_mask: torch.Tensor,
    causal_mask: torch.Tensor,
    logits_base: torch.Tensor,
    position_bias: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    bucket: torch.Tensor,
    fill_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_bias = torch.where(
        causal_mask.expand(BATCH, 1, logits_base.shape[-2], logits_base.shape[-1]),
        fill_value,
        torch.tensor(NEG_INF_F32, dtype=logits_base.dtype, device=logits_base.device),
    )
    logits = logits_base.view(BATCH, HEADS, logits_base.shape[-2], logits_base.shape[-1])
    bias = position_bias.permute(2, 0, 1).unsqueeze(0)
    softmax_probs = torch.exp(logits + bias + mask_bias - row_max) / row_sum
    dscore = softmax_backward_from_probs(bmm_grad.view_as(softmax_probs), softmax_probs, keep_mask)
    attention_grad = residuals[0] + residuals[1] + residuals[2] + residuals[3] + residuals[4] + dscore
    bucket_grad = relative_position_bias_grad(attention_grad, bucket, fill_value)
    return dscore.reshape_as(bmm_grad), bucket_grad


def torch_direct_oracle(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        view_40,
        view_87,
        view_134,
        view_181,
        view_228,
        bmm_45,
        arg226_1,
        arg225_1,
        arg224_1,
        full_1,
        view_302,
        view_328,
        view_354,
        view_380,
        view_406,
        bmm_69,
        arg141_1,
        arg132_1,
        arg136_1,
        arg138_1,
        arg139_1,
        arg140_1,
        arg137_1,
        *_shape_params,
    ) = inputs

    dscore_0, bucket_grad_0 = first_attention_backward(
        (view_40, view_87, view_134, view_181, view_228),
        bmm_45,
        arg226_1,
        arg225_1,
        arg224_1,
        full_1,
    )
    dscore_1, bucket_grad_1 = second_attention_backward(
        (view_302, view_328, view_354, view_380, view_406),
        bmm_69,
        arg141_1,
        arg132_1,
        arg136_1,
        arg138_1,
        arg139_1,
        arg140_1,
        arg137_1,
        full_1,
    )
    return dscore_0, bucket_grad_0, dscore_1, bucket_grad_1


def triton_fused_attention_backward_oracle(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Placeholder for the intended fused Triton implementation.

    TODO for the exact oracle floor:
    1. Implement a row-wise softmax-backward kernel over ``B*H*Q`` rows and K
       tiles.  Load ``dprob``, boolean dropout/keep mask, and either stored
       softmax probabilities or reconstructed probabilities from logits,
       relative bias, mask bias, row max, and row sum.
    2. Fuse ``dscore = p * (dprob - sum(dprob * p))`` with the five residual
       adds so the attention-score gradient is written once.  The kernel should
       optionally expose the raw ``dscore`` output because the captured repro
       returns the reshaped ``[64, Q, K]`` tensor before the residual add.
    3. Accumulate relative-position-bias gradients without materializing the
       full ``[Q, K, H]`` permute.  Candidate designs are per-head/block
       partial bucket histograms followed by a second reduction, or atomics into
       ``[32, 8]`` with deterministic/reference mode for correctness checks.
    4. Handle ``bucket == -1`` exactly like the repro: values are replaced with
       ``full_1`` before the accumulated ``index_put``.  Do not treat ``-1`` as
       a sentinel to skip unless the replacement value is known to be zero.
    5. Validate against ``torch_direct_oracle`` at small ``--seq-len`` first,
       then run default ``--seq-len 1024`` on CUDA before appending measured
       floors.

    The direct PyTorch oracle remains runnable today for CLI correctness,
    benchmarking scaffolding, and measured_oracle_floors.csv append plumbing.
    """
    raise NotImplementedError(
        "Triton fused softmax-backward/attention-backward oracle is a TODO; "
        "use --impl torch-direct for the runnable scaffold."
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], tuple[torch.Tensor, ...]], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / rep * 1_000_000.0


def max_abs_diff(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> float:
    return max((a.float() - e.float()).abs().max().item() for a, e in zip(actual, expected))


def allclose(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], rtol: float, atol: float) -> bool:
    return all(torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol) for a, e in zip(actual, expected))


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_baseline_row() -> dict[str, str]:
    path = REPO_ROOT / "investigation_results" / "sol_gap_candidates.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("repro_id") == REPRO_ID:
                return row
    return {}


def append_csv(
    path: Path,
    impl: str,
    device: torch.device,
    oracle_us: float,
    correct: str,
    diff: float,
    args: argparse.Namespace,
) -> None:
    baseline = load_baseline_row()
    best_compile_us = float(baseline.get("best_compile_us", "nan"))
    memcopy_sol_us = float(baseline.get("memcopy_sol_us", "nan"))
    total_bytes = int(float(baseline.get("total_bytes", "0") or 0))
    n_kernels = int(float(baseline.get("n_kernels", "0") or 0))

    row = {
        "repro_id": REPRO_ID,
        "repro_path": str(REPRO_PATH.relative_to(REPO_ROOT)),
        "shape_label": f"8x8x{args.seq_len}x{args.seq_len}",
        "family": "fused_softmax_backward_attention_backward",
        "oracle_impl": impl,
        "oracle_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "hardware": args.hardware,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
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
        "max_abs_diff": diff,
        "tolerance": f"rtol={args.rtol},atol={args.atol}" if args.check else "not_checked",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "Torch direct softmax-backward scaffold; see Triton TODO for fused attention-backward and bucket accumulation plan.",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("torch-direct", "triton-todo"), default="torch-direct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=16, help="Use 1024 for the captured default shape.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="Compare oracle output to captured repro.py on CUDA.")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--no-bench", action="store_true", help="Skip timing and CSV append.")
    parser.add_argument("--no-append", action="store_true", help="Do not append measured_oracle_floors.csv.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--hardware", default="unknown")
    return parser.parse_args()


def oracle_forward(inputs):
    return triton_fused_attention_backward_oracle(*inputs)


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
