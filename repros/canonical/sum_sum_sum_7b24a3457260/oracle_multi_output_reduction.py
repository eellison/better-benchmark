"""
Canonical-local oracle scaffold for repro sum_sum_sum_7b24a3457260.

Repro pattern:
    Two T5 attention-backward-like branches that each return a reshaped
    softmax-gradient tensor and an index-accumulated projection-gradient tensor.
    The captured graph has four sum reductions total: two reductions over the
    softmax key dimension and two reductions over batch.

This file is intentionally local to the canonical repro so candidate one-pass
multi-output reduction kernels can be developed and measured without changing
the captured repro.  The runnable implementation below is a direct PyTorch
oracle/reference that keeps the multi-output reduction structure explicit while
sharing the branch-local expressions.  The Triton entry point is a scaffold with
TODOs for an exact fused reduction implementation.
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_7b24a3457260"
SHAPE_LABEL = "torchbench_hf_t5_base_train_001_7966e243"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
SCALE = 1.1111111111111112


def index_accumulate(values_nhwd: torch.Tensor, indices_hw: torch.Tensor, fill_value: torch.Tensor) -> torch.Tensor:
    masked_values = torch.where(indices_hw.eq(-1).unsqueeze(-1), fill_value, values_nhwd).clone(
        memory_format=torch.contiguous_format
    )
    out = torch.full((32, 12), 0.0, dtype=values_nhwd.dtype, device=values_nhwd.device)
    return out.index_put((indices_hw,), masked_values, accumulate=True)


def attention_branch_from_probs(
    residual_terms: tuple[torch.Tensor, ...],
    bmm: torch.Tensor,
    dropout_mask: torch.Tensor,
    probs: torch.Tensor,
    indices: torch.Tensor,
    fill_value: torch.Tensor,
    bmm_shape,
    flat_shape,
    roundtrip_shape,
    out_shape,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_sum = residual_terms[0]
    for term in residual_terms[1:]:
        residual_sum = residual_sum + term

    dropped = bmm.view(bmm_shape) * dropout_mask.to(torch.float32) * SCALE
    weighted = dropped * probs
    key_sum = weighted.sum(dim=-1, keepdim=True)
    softmax_grad = (-probs) * key_sum + weighted

    flat_grad = softmax_grad.view(flat_shape)
    roundtrip_grad = flat_grad.view(roundtrip_shape)
    out_grad = roundtrip_grad.view(out_shape)

    batch_sum = (residual_sum + roundtrip_grad).sum(dim=0, keepdim=True).squeeze(0)
    projected = batch_sum.permute(1, 2, 0)
    indexed = index_accumulate(projected, indices, fill_value)
    return out_grad, indexed


def attention_branch_from_logits(
    residual_terms: tuple[torch.Tensor, ...],
    bmm: torch.Tensor,
    dropout_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    scores: torch.Tensor,
    bias: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    indices: torch.Tensor,
    fill_value: torch.Tensor,
    bmm_shape,
    mask_shape,
    scores_shape,
    flat_shape,
    roundtrip_shape,
    out_shape,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_mask = attn_mask.expand(mask_shape)
    neg_inf = torch.full((), torch.finfo(torch.float32).min, dtype=torch.float32, device=fill_value.device)
    masked = torch.where(expanded_mask, fill_value, neg_inf)
    logits = scores.view(scores_shape) + bias.permute(2, 0, 1).unsqueeze(0) + masked
    probs = torch.exp(logits - row_max) / row_sum

    dropped = bmm.view(bmm_shape) * dropout_mask.to(torch.float32) * SCALE
    weighted = dropped * probs
    key_sum = weighted.sum(dim=-1, keepdim=True)
    softmax_grad = (-probs) * key_sum + weighted

    flat_grad = softmax_grad.view(flat_shape)
    roundtrip_grad = flat_grad.view(roundtrip_shape)
    out_grad = roundtrip_grad.view(out_shape)

    residual_sum = residual_terms[0]
    for term in residual_terms[1:]:
        residual_sum = residual_sum + term
    batch_sum = (residual_sum + roundtrip_grad).sum(dim=0, keepdim=True).squeeze(0)
    projected = batch_sum.permute(1, 2, 0)
    indexed = index_accumulate(projected, indices, fill_value)
    return out_grad, indexed


def torch_direct_oracle(
    view_40: torch.Tensor,
    view_87: torch.Tensor,
    view_134: torch.Tensor,
    view_181: torch.Tensor,
    view_228: torch.Tensor,
    view_275: torch.Tensor,
    view_322: torch.Tensor,
    view_369: torch.Tensor,
    view_416: torch.Tensor,
    view_463: torch.Tensor,
    view_510: torch.Tensor,
    bmm_93: torch.Tensor,
    arg430_1: torch.Tensor,
    arg429_1: torch.Tensor,
    arg428_1: torch.Tensor,
    full_1: torch.Tensor,
    view_584: torch.Tensor,
    view_610: torch.Tensor,
    view_636: torch.Tensor,
    view_662: torch.Tensor,
    view_688: torch.Tensor,
    view_714: torch.Tensor,
    view_740: torch.Tensor,
    view_766: torch.Tensor,
    view_792: torch.Tensor,
    view_818: torch.Tensor,
    view_844: torch.Tensor,
    bmm_141: torch.Tensor,
    arg267_1: torch.Tensor,
    arg258_1: torch.Tensor,
    arg262_1: torch.Tensor,
    arg264_1: torch.Tensor,
    arg265_1: torch.Tensor,
    arg266_1: torch.Tensor,
    arg263_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
    _shape_param_9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct PyTorch formulation of the four captured outputs."""
    first = attention_branch_from_probs(
        (
            view_40,
            view_87,
            view_134,
            view_181,
            view_228,
            view_275,
            view_322,
            view_369,
            view_416,
            view_463,
            view_510,
        ),
        bmm_93,
        arg430_1,
        arg429_1,
        arg428_1,
        full_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )

    second = attention_branch_from_logits(
        (
            view_584,
            view_610,
            view_636,
            view_662,
            view_688,
            view_714,
            view_740,
            view_766,
            view_792,
            view_818,
            view_844,
        ),
        bmm_141,
        arg267_1,
        arg258_1,
        arg262_1,
        arg264_1,
        arg265_1,
        arg266_1,
        arg263_1,
        full_1,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
    )

    return first[0], first[1], second[0], second[1]


def triton_one_pass_oracle(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Placeholder for the intended one-pass Triton implementation.

    TODO for P0 multi_output_reduction_templates:
    1. Fuse each attention branch around a 2-D grid over the flattened
       B*H*Q rows and K columns.  The row program should load bmm/dropout and
       either existing probabilities or logits+mask+bias, then compute the
       softmax-gradient row reduction ``sum_k(dropped * probs)``.
    2. Return the reshaped softmax-gradient tensor directly while also feeding
       a second reduction over batch for the projection-gradient output.
    3. Implement the projection-gradient as an indexed accumulation into
       ``[32, 12]``.  Exact parity with ``aten.index_put(..., accumulate=True)``
       requires preserving negative-index behavior after the repro's sentinel
       ``where(index == -1, full_1, value)`` masking.
    4. Share loads and arithmetic between the two outputs in each branch, and
       evaluate whether both branches can be scheduled together or should use
       two branch-specialized kernels plus a tiny accumulation kernel.
    5. Validate against ``torch_direct_oracle`` with fp32 tolerances before
       recording measured floors.

    The direct Torch oracle remains runnable so this scaffold can be used for
    CLI correctness, benchmarking structure, and CSV append plumbing today.
    """
    raise NotImplementedError(
        "Triton one-pass multi-output reduction kernel is a TODO; "
        "use --impl torch-direct for the runnable scaffold."
    )


def make_inputs(device: torch.device) -> tuple:
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


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
        "shape_label": SHAPE_LABEL,
        "family": "multi_output_reduction",
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
        "notes": "Torch direct T5 attention multi-output reduction scaffold; see Triton TODO.",
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
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="Compare oracle output to captured repro.py.")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--no-bench", action="store_true", help="Skip timing and CSV append.")
    parser.add_argument("--no-append", action="store_true", help="Do not append measured_oracle_floors.csv.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--hardware", default="unknown")
    return parser.parse_args()


@oracle_impl(hardware="H100", shapes="(T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([96, 1024, 1024], f32), T([8, 12, 1024, 1024], b8), T([8, 12, 1024, 1024], f32), T([1024, 1024], i64, gen=Index(32)), T([], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([96, 1024, 1024], f32), T([8, 12, 1024, 1024], b8), T([1, 1, 1024, 1], b8), T([96, 1024, 1024], f32), T([1024, 1024, 12], f32), T([8, 12, 1024, 1], f32), T([8, 12, 1024, 1], f32), T([1024, 1024], i64, gen=Index(32)), S([8, 12, 1024, 1024]), S([96, 1024, 1024]), S([8, 12, 1024, 1024]), S([96, 1024, 1024]), S([8, 12, 1024, 1024]), S([8, -1, 1024, 1024]), S([8, 12, 1024, 1024]), S([96, 1024, 1024]), S([8, 12, 1024, 1024]), S([96, 1024, 1024]))")
def oracle_forward(inputs):
    return triton_one_pass_oracle(*inputs)


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
