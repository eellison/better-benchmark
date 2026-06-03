"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full ignore-index cross-entropy mean returned by Repro.forward, including reshape, row logsumexp, masked gather, valid-count reduction, loss reduction, and final division in Triton kernels, whereas Inductor currently lowers the view/amax/sub/exp/sum/log/sub/gather/where/sum/count/div graph as separate generic row reductions, pointwise kernels, and scalar reductions that materialize the full log-softmax intermediate; Inductor cannot do this today because its scheduler/codegen pattern library only recognizes narrower online softmax fragments and does not canonicalize ignore-index cross-entropy mean with both loss and valid-count reductions into a fused row-reduction plus scalar epilogue; the fix is NEW_PATTERN: add an Inductor pattern/lowering for log_softmax plus masked gather plus loss/count mean that emits an online cross-entropy row kernel and a small final reduction directly."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_sum_0e37ca9164b3"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

DEFAULT_BATCH = 32
DEFAULT_SEQ_LEN = 128
DEFAULT_N_COLS = 8008


@triton.jit
def _xent_rows_kernel(
    labels_ptr,
    logits_ptr,
    loss_ptr,
    valid_ptr,
    n_cols: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * n_cols

    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)
    target_logit = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
    ).to(tl.float32)

    row_max = tl.full([], -float("inf"), tl.float32)
    denom = tl.full([], 0.0, tl.float32)

    for block_start in tl.range(0, n_cols, block_n):
        cols = block_start + tl.arange(0, block_n)
        mask = cols < n_cols
        logits = tl.load(
            logits_ptr + row_start + cols,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(row_max, block_max)
        denom = denom * tl.exp(row_max - new_max) + tl.sum(
            tl.exp(logits - new_max),
            axis=0,
        )
        row_max = new_max

    loss = row_max + tl.log(denom) - target_logit
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


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _check_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def _effective_block_n(block_n: int, n_cols: int) -> int:
    _check_power_of_two(block_n, "block_n")
    return min(block_n, triton.next_power_of_2(n_cols))


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_synthetic_inputs(
    batch: int,
    seq_len: int,
    n_cols: int,
    seed: int,
    ignore_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    if batch <= 0 or seq_len <= 0 or n_cols <= 0:
        raise ValueError("batch, seq_len, and cols must all be positive")
    if not 0.0 <= ignore_frac <= 1.0:
        raise ValueError(f"ignore_frac must be in [0, 1], got {ignore_frac}")

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    logits = torch.randn(
        (batch * seq_len, n_cols),
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
    labels = torch.randint(
        0,
        n_cols,
        (batch, seq_len),
        device="cuda",
        dtype=torch.int64,
        generator=gen,
    )
    if ignore_frac > 0.0:
        ignore = torch.rand((batch, seq_len), device="cuda", generator=gen) < ignore_frac
        labels = torch.where(ignore, torch.full_like(labels, -100), labels)
        labels.view(-1)[0] = 0
    return logits, labels, [batch, seq_len, n_cols], [-1, n_cols]


def _launch_oracle(
    labels_1d: torch.Tensor,
    logits_2d: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert labels_1d.is_cuda and logits_2d.is_cuda
    assert labels_1d.dtype == torch.int64 and labels_1d.ndim == 1
    assert logits_2d.dtype == torch.float32 and logits_2d.ndim == 2

    n_rows, n_cols = logits_2d.shape
    assert labels_1d.shape == (n_rows,)
    assert loss_per_row.shape == (n_rows,) and loss_per_row.dtype == torch.float32
    assert valid_per_row.shape == (n_rows,) and valid_per_row.dtype == torch.float32
    assert out.shape == () and out.dtype == torch.float32

    effective_block = _effective_block_n(block_n, n_cols)
    _xent_rows_kernel[(n_rows,)](
        labels_1d,
        logits_2d,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        block_n=effective_block,
        num_warps=num_warps,
    )

    block_m = triton.next_power_of_2(n_rows)
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        out,
        n_rows=n_rows,
        block_m=block_m,
        num_warps=8,
    )
    return out


def oracle_ignore_index_cross_entropy_mean(
    mm: torch.Tensor,
    labels: torch.Tensor,
    shape_3d: list[int],
    shape_2d: list[int],
    block_n: int = 8192,
    num_warps: int = 16,
) -> torch.Tensor:
    logits_2d = mm.view(shape_3d).view(shape_2d)
    labels_1d = labels.view(-1)
    n_rows = logits_2d.shape[0]

    loss_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    out = torch.empty((), device=mm.device, dtype=torch.float32)
    return _launch_oracle(
        labels_1d,
        logits_2d,
        loss_per_row,
        valid_per_row,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _compare_outputs(ref: Any, got: Any, rtol: float, atol: float) -> bool:
    ref_items = _as_tuple(ref)
    got_items = _as_tuple(got)
    if len(ref_items) != len(got_items):
        print(f"output arity mismatch: ref={len(ref_items)} oracle={len(got_items)}")
        return False

    ok = True
    for idx, (ref_item, got_item) in enumerate(zip(ref_items, got_items)):
        if isinstance(ref_item, torch.Tensor) and isinstance(got_item, torch.Tensor):
            if ref_item.shape != got_item.shape or ref_item.dtype != got_item.dtype:
                print(
                    f"output[{idx}] metadata mismatch: "
                    f"ref shape={tuple(ref_item.shape)} dtype={ref_item.dtype}, "
                    f"oracle shape={tuple(got_item.shape)} dtype={got_item.dtype}"
                )
                ok = False
                continue

            ref_f = ref_item.float()
            got_f = got_item.float()
            diff = (ref_f - got_f).abs()
            finite_diff = diff[torch.isfinite(diff)]
            max_abs = finite_diff.max().item() if finite_diff.numel() else float("nan")
            denom = ref_f.abs().clamp_min(1e-8)
            rel = diff / denom
            finite_rel = rel[torch.isfinite(rel)]
            max_rel = finite_rel.max().item() if finite_rel.numel() else float("nan")
            item_ok = torch.allclose(ref_f, got_f, rtol=rtol, atol=atol, equal_nan=True)
            print(
                f"output[{idx}]: shape={list(ref_item.shape)} dtype={ref_item.dtype} "
                f"ref={ref_item.detach().flatten()[:3].tolist()} "
                f"oracle={got_item.detach().flatten()[:3].tolist()} "
                f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={item_ok}"
            )
            ok = ok and bool(item_ok)
        else:
            item_ok = ref_item == got_item
            print(f"output[{idx}]: ref={ref_item!r} oracle={got_item!r} equal={item_ok}")
            ok = ok and bool(item_ok)
    return ok


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_check(args: argparse.Namespace) -> bool:
    _require_cuda()
    module = _load_repro_module()
    inputs = _make_synthetic_inputs(
        batch=args.batch,
        seq_len=args.seq_len,
        n_cols=args.cols,
        seed=args.seed,
        ignore_frac=args.ignore_frac,
    )
    logits, labels, shape_3d, shape_2d = inputs
    model = module.Repro().cuda()

    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_ignore_index_cross_entropy_mean(
            logits,
            labels,
            shape_3d,
            shape_2d,
            block_n=args.block_n,
            num_warps=args.num_warps,
        )
        torch.cuda.synchronize()

    ignore_count = (labels == -100).sum().item()
    print(
        f"check ignore-index xent mean: logits=f32[{logits.shape[0]}, {logits.shape[1]}] "
        f"labels=i64[{labels.shape[0]}, {labels.shape[1]}] ignore_count={ignore_count}"
    )
    ok = _compare_outputs(ref, got, rtol=args.rtol, atol=args.atol)
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(args: argparse.Namespace) -> None:
    _require_cuda()
    logits, labels, shape_3d, shape_2d = _make_synthetic_inputs(
        batch=args.batch,
        seq_len=args.seq_len,
        n_cols=args.cols,
        seed=args.seed,
        ignore_frac=args.ignore_frac,
    )
    logits_2d = logits.view(shape_3d).view(shape_2d)
    labels_1d = labels.view(-1)
    n_rows, n_cols = logits_2d.shape
    loss_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)

    read_bytes = n_rows * n_cols * 4 + n_rows * (8 + 4 + 8)
    write_bytes = n_rows * 8 + 4
    total_bytes = read_bytes + write_bytes
    print(
        f"bench ignore-index xent mean: logits=f32[{n_rows}, {n_cols}] "
        f"labels=i64[{labels.shape[0]}, {labels.shape[1]}]"
    )
    print(f"logical oracle traffic={total_bytes / 1e6:.3f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                labels_1d,
                logits_2d,
                loss_per_row,
                valid_per_row,
                out,
                block_n=args.block_n,
                num_warps=args.num_warps,
            ),
            warmup=args.warmup,
            rep=args.rep,
        )
    bandwidth = total_bytes / (oracle_us * 1e-6) / 1e12
    print(f"oracle_us={oracle_us:.3f} bandwidth={bandwidth:.3f} TB/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare oracle against the full Repro.forward return value")
    parser.add_argument("--bench", action="store_true", help="benchmark the Triton oracle")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--cols", type=int, default=DEFAULT_N_COLS)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ignore-frac", type=float, default=0.05)
    parser.add_argument("--block-n", type=int, default=8192)
    parser.add_argument("--num-warps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("choose at least one explicit mode: --check or --bench")

    if args.check and not run_check(args):
        sys.exit(1)
    if args.bench:
        run_bench(args)


if __name__ == "__main__":
    with torch.no_grad():
        main()
