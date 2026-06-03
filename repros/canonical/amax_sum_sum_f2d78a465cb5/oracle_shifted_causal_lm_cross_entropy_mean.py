"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full GPT-J shifted-label causal-LM cross-entropy mean returned by repro.py by scanning each logits row once with scalar online max and sum-exp accumulators, loading the shifted target logit, accumulating masked loss and valid count, and emitting the scalar mean, whereas Inductor lowers the pad/slice/ne/log-softmax/gather/masked-sum/count/div graph as separate generic reductions and pointwise kernels that materialize row log-softmax data; Inductor cannot do this today because its online softmax scheduler/template matcher does not recognize the full shifted ignore-index cross-entropy mean with gather, valid-count reduction, and final division as one fused row-reduction pattern; the fix is NEW_PATTERN: add an Inductor cross-entropy-mean pattern that fuses shifted target construction, online row logsumexp, target gather, masked loss/count accumulation, and scalar mean codegen."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_sum_f2d78a465cb5"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

DEFAULT_BATCH = 1
DEFAULT_SEQ_LEN = 128
DEFAULT_N_COLS = 50400


@triton.jit
def _online_shifted_xent_rows_kernel(
    tokens_ptr,
    logits_ptr,
    loss_ptr,
    valid_ptr,
    n_cols: tl.constexpr,
    seq_len: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    pos = row % seq_len
    is_last_token = pos == (seq_len - 1)

    label = tl.load(tokens_ptr + row + 1, mask=~is_last_token, other=-100)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)

    row_start = row * n_cols
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
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    tokens = torch.randint(
        0,
        n_cols,
        (batch, seq_len),
        device="cuda",
        dtype=torch.int64,
        generator=gen,
    )
    if seq_len > 1:
        ignore = torch.rand((batch, seq_len), device="cuda", generator=gen) < 0.05
        tokens = torch.where(ignore, torch.full_like(tokens, -100), tokens)
        tokens[:, 1] = torch.clamp(tokens[:, 1], min=0)

    logits = torch.randn(
        (batch * seq_len, n_cols),
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
    return tokens, logits, [batch, seq_len, n_cols], [-1, n_cols]


def _make_repro_inputs(
    module: Any,
    batch: int,
    seq_len: int,
    n_cols: int,
    seed: int,
) -> tuple[Any, ...]:
    if (batch, seq_len, n_cols) == (DEFAULT_BATCH, DEFAULT_SEQ_LEN, DEFAULT_N_COLS):
        return tuple(
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in module.make_inputs()
        )
    return _make_synthetic_inputs(batch, seq_len, n_cols, seed)


def _launch_oracle(
    tokens: torch.Tensor,
    logits_2d: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert tokens.is_cuda and logits_2d.is_cuda
    assert tokens.dtype == torch.int64
    assert logits_2d.dtype == torch.float32 and logits_2d.ndim == 2
    assert tokens.ndim == 2

    batch, seq_len = tokens.shape
    n_rows, n_cols = logits_2d.shape
    assert n_rows == batch * seq_len
    assert loss_per_row.shape == (n_rows,) and loss_per_row.dtype == torch.float32
    assert valid_per_row.shape == (n_rows,) and valid_per_row.dtype == torch.float32
    assert out.shape == () and out.dtype == torch.float32

    effective_block = _effective_block_n(block_n, n_cols)
    _online_shifted_xent_rows_kernel[(n_rows,)](
        tokens,
        logits_2d,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        seq_len=seq_len,
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


def oracle_shifted_causal_lm_cross_entropy_mean(
    tokens: torch.Tensor,
    logits_2d: torch.Tensor,
    block_n: int = 4096,
    num_warps: int = 8,
) -> torch.Tensor:
    n_rows = logits_2d.shape[0]
    loss_per_row = torch.empty((n_rows,), device=logits_2d.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=logits_2d.device, dtype=torch.float32)
    out = torch.empty((), device=logits_2d.device, dtype=torch.float32)
    return _launch_oracle(
        tokens,
        logits_2d,
        loss_per_row,
        valid_per_row,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


def oracle_from_repro_inputs(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    shape_3d: list[int],
    shape_2d: list[int],
    block_n: int = 4096,
    num_warps: int = 8,
) -> torch.Tensor:
    logits_2d = logits.view(shape_3d).view(shape_2d)
    return oracle_shifted_causal_lm_cross_entropy_mean(
        tokens,
        logits_2d,
        block_n=block_n,
        num_warps=num_warps,
    )


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _compare_outputs(
    actual: Any,
    expected: Any,
    rtol: float,
    atol: float,
) -> bool:
    actual_tuple = _as_tuple(actual)
    expected_tuple = _as_tuple(expected)
    if len(actual_tuple) != len(expected_tuple):
        print(f"output_count_mismatch actual={len(actual_tuple)} expected={len(expected_tuple)}")
        return False

    all_ok = True
    for idx, (got, ref) in enumerate(zip(actual_tuple, expected_tuple)):
        if not isinstance(got, torch.Tensor) or not isinstance(ref, torch.Tensor):
            ok = got == ref
            print(f"output[{idx}]: non_tensor allclose={ok}")
            all_ok = all_ok and bool(ok)
            continue

        got_f = got.float()
        ref_f = ref.float()
        diff = (got_f - ref_f).abs()
        finite_diff = diff[torch.isfinite(diff)]
        max_abs = finite_diff.max().item() if finite_diff.numel() else float("nan")
        denom = ref_f.abs().clamp_min(1e-8)
        rel = diff / denom
        finite_rel = rel[torch.isfinite(rel)]
        max_rel = finite_rel.max().item() if finite_rel.numel() else float("nan")
        ok = torch.allclose(got_f, ref_f, rtol=rtol, atol=atol, equal_nan=True)
        print(
            f"output[{idx}]: shape={list(ref.shape)} dtype={ref.dtype} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={ok}"
        )
        all_ok = all_ok and bool(ok)
    return all_ok


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


def run_check(
    batch: int,
    seq_len: int,
    n_cols: int,
    block_n: int,
    num_warps: int,
    rtol: float,
    atol: float,
) -> bool:
    _require_cuda()
    module = _load_repro_module()
    inputs = _make_repro_inputs(module, batch, seq_len, n_cols, seed=1234)

    with torch.no_grad():
        repro_out = module.Repro().cuda()(*inputs)
        oracle_out = oracle_from_repro_inputs(
            *inputs,
            block_n=block_n,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

    print(
        f"check shifted causal-lm xent mean: tokens=i64[{batch}, {seq_len}] "
        f"logits=f32[{batch * seq_len}, {n_cols}]"
    )
    ok = _compare_outputs(oracle_out, repro_out, rtol=rtol, atol=atol)
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(
    batch: int,
    seq_len: int,
    n_cols: int,
    block_n: int,
    num_warps: int,
    warmup: int,
    rep: int,
    include_eager: bool,
) -> None:
    _require_cuda()
    module = _load_repro_module()
    inputs = _make_repro_inputs(module, batch, seq_len, n_cols, seed=4321)
    tokens, logits, shape_3d, shape_2d = inputs
    logits_2d = logits.view(shape_3d).view(shape_2d)
    n_rows = batch * seq_len

    loss_per_row = torch.empty((n_rows,), device="cuda", dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device="cuda", dtype=torch.float32)
    out = torch.empty((), device="cuda", dtype=torch.float32)

    read_bytes = n_rows * n_cols * 4 + n_rows * (8 + 4)
    write_bytes = n_rows * 8 + 4
    total_bytes = read_bytes + write_bytes

    print(f"bench shifted causal-lm xent mean: tokens=i64[{batch}, {seq_len}]")
    print(f"logits=f32[{n_rows}, {n_cols}], logical oracle traffic={total_bytes / 1e6:.2f} MB")
    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                tokens,
                logits_2d,
                loss_per_row,
                valid_per_row,
                out,
                block_n=block_n,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )
    bandwidth = total_bytes / (oracle_us * 1e-6) / 1e12
    print(f"oracle_us={oracle_us:.3f} bandwidth={bandwidth:.3f} TB/s")

    if include_eager:
        model = module.Repro().cuda()
        with torch.no_grad():
            eager_us = _bench_cuda(lambda: model(*inputs), warmup=warmup, rep=rep)
        print(f"repro_eager_us={eager_us:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare oracle against the full repro.py return value")
    parser.add_argument("--bench", action="store_true", help="benchmark the Triton oracle")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--cols", type=int, default=DEFAULT_N_COLS)
    parser.add_argument("--block-n", type=int, default=4096)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--include-eager", action="store_true", help="also time eager repro.py in --bench mode")
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select --check and/or --bench")

    if args.check:
        ok = run_check(
            batch=args.batch,
            seq_len=args.seq_len,
            n_cols=args.cols,
            block_n=args.block_n,
            num_warps=args.num_warps,
            rtol=args.rtol,
            atol=args.atol,
        )
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            batch=args.batch,
            seq_len=args.seq_len,
            n_cols=args.cols,
            block_n=args.block_n,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            include_eager=args.include_eager,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
