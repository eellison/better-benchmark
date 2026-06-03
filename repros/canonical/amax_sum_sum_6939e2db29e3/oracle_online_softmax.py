"""
Gap diagnosis (classification: NEW_PATTERN): this oracle computes the shifted-label cross-entropy mean by reading each logits row once with scalar online max and denominator accumulators plus one target-logit load, then reducing only per-row losses and valid-counts; Inductor lowers the decomposed amax/sub/exp/sum/log/log_softmax/gather/mask/mean graph as generic reductions and pointwise work that materializes and rereads the full log-softmax-sized intermediate, because today it does not canonicalize this shifted ignore-index cross-entropy idiom into an online softmax row template with a scalar epilogue and sibling valid-count reduction; the Inductor fix is a NEW_PATTERN lowering for log_softmax+gather+masked-mean cross entropy that emits this online accumulator kernel and the small final scalar reduction directly.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_sum_6939e2db29e3"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ_LEN = 512
N_COLS = 29056
M_ROWS = BATCH * SEQ_LEN

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


@triton.jit
def _online_xent_rows_kernel(
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
    x_label = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
    ).to(tl.float32)

    row_max = tl.full([], -float("inf"), tl.float32)
    denom = tl.full([], 0.0, tl.float32)

    for block_start in tl.range(0, n_cols, block_n):
        cols = block_start + tl.arange(0, block_n)
        col_mask = cols < n_cols
        x = tl.load(
            logits_ptr + row_start + cols,
            mask=col_mask,
            other=-float("inf"),
        ).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        denom = denom * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    loss = row_max + tl.log(denom) - x_label
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


def _shifted_labels(tokens: torch.Tensor) -> torch.Tensor:
    labels = torch.empty_like(tokens)
    labels[:, :-1] = tokens[:, 1:]
    labels[:, -1] = -100
    return labels.reshape(-1)


def eager_reference(tokens: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    labels = _shifted_labels(tokens)
    valid = labels != -100
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    row_max = logits.amax(dim=1, keepdim=True)
    shifted = logits - row_max
    log_denom = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    log_probs = shifted - log_denom

    gathered = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    losses = torch.where(valid, -gathered, torch.zeros_like(gathered))
    return losses.sum() / valid.sum().to(torch.float32)


def _launch_oracle(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert tokens.is_cuda and logits.is_cuda
    assert tokens.dtype == torch.int64
    assert logits.dtype == torch.float32 and logits.ndim == 2
    assert tokens.ndim == 2
    n_rows, n_cols = logits.shape
    batch, seq_len = tokens.shape
    assert n_rows == batch * seq_len
    assert loss_per_row.shape == (n_rows,) and loss_per_row.dtype == torch.float32
    assert valid_per_row.shape == (n_rows,) and valid_per_row.dtype == torch.float32
    assert out.shape == () and out.dtype == torch.float32

    _online_xent_rows_kernel[(n_rows,)](
        tokens,
        logits,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        seq_len=seq_len,
        block_n=block_n,
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


def oracle_online_softmax_xent_mean(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    block_n: int = 4096,
    num_warps: int = 8,
) -> torch.Tensor:
    n_rows = logits.shape[0]
    loss_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    return _launch_oracle(
        tokens,
        logits,
        loss_per_row,
        valid_per_row,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


def _make_inputs(
    batch: int,
    seq_len: int,
    n_cols: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    ignore = torch.rand((batch, seq_len), device="cuda", generator=gen) < 0.05
    tokens = torch.where(ignore, torch.full_like(tokens, -100), tokens)
    tokens[:, 0] = torch.clamp(tokens[:, 0], min=0)
    logits = torch.randn(
        (batch * seq_len, n_cols),
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
    return tokens, logits


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _compile_with_config(
    module,
    inputs: tuple,
    config: dict[str, object],
    warmup: int,
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_check(
    batch: int,
    seq_len: int,
    n_cols: int,
    block_n: int,
    num_warps: int,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    tokens, logits = _make_inputs(batch, seq_len, n_cols, seed=1234)
    with torch.no_grad():
        ref = eager_reference(tokens, logits)
        got = oracle_online_softmax_xent_mean(
            tokens,
            logits,
            block_n=block_n,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

    abs_diff = (got - ref).abs().item()
    rel_diff = (got - ref).abs().div(ref.abs().clamp_min(1e-8)).item()
    ok = torch.allclose(got, ref, rtol=5e-5, atol=5e-5)

    print(
        f"check online xent mean: shape=({batch}, {seq_len}, {n_cols}) "
        f"ref={ref.item():.8f} oracle={got.item():.8f} "
        f"max_abs={abs_diff:.6e} max_rel={rel_diff:.6e} allclose={ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _compile_inputs(
    module,
    batch: int,
    seq_len: int,
    n_cols: int,
    seed: int,
) -> tuple:
    if (batch, seq_len, n_cols) == (BATCH, SEQ_LEN, N_COLS):
        inputs = module.make_inputs()
        return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    tokens, logits = _make_inputs(batch, seq_len, n_cols, seed=seed)
    return logits, tokens, [batch, seq_len, n_cols], [-1, n_cols]


def run_bench(
    batch: int,
    seq_len: int,
    n_cols: int,
    block_n: int,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    tokens, logits = _make_inputs(batch, seq_len, n_cols, seed=4321)
    n_rows = batch * seq_len
    loss_per_row = torch.empty((n_rows,), device="cuda", dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device="cuda", dtype=torch.float32)
    out = torch.empty((), device="cuda", dtype=torch.float32)

    read_bytes = n_rows * n_cols * 4 + n_rows * 8
    write_bytes = n_rows * 8 + 4
    total_bytes = read_bytes + write_bytes

    print(f"oracle shape: tokens=i64[{batch}, {seq_len}], logits=f32[{n_rows}, {n_cols}]")
    print(f"single-pass logical traffic: {total_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                tokens,
                logits,
                loss_per_row,
                valid_per_row,
                out,
                block_n=block_n,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = total_bytes / (oracle_us * 1e-6) / 1e12
    print(f"oracle online xent mean: {oracle_us:.3f} us ({oracle_bw:.3f} TB/s)")

    if no_compile:
        return

    module = _load_repro_module()
    compile_inputs = _compile_inputs(module, batch, seq_len, n_cols, seed=4321)
    print("torch.compile full repro timings include label shift and valid-count reduction")

    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module, compile_inputs, config, warmup)
            us = _bench_cuda(lambda: compiled(*compile_inputs), warmup=warmup, rep=rep)
            print(f"torch.compile {label}: {us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="benchmark repetitions")
    parser.add_argument("--block-n", type=int, default=4096, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=8, help="Triton warps for row kernel")
    parser.add_argument("--check-batch", type=int, default=4)
    parser.add_argument("--check-seq", type=int, default=16)
    parser.add_argument("--check-cols", type=int, default=1024)
    parser.add_argument("--bench-batch", type=int, default=BATCH)
    parser.add_argument("--bench-seq", type=int, default=SEQ_LEN)
    parser.add_argument("--bench-cols", type=int, default=N_COLS)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check(
            batch=args.check_batch,
            seq_len=args.check_seq,
            n_cols=args.check_cols,
            block_n=min(args.block_n, triton.next_power_of_2(args.check_cols)),
            num_warps=args.num_warps,
        )
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            batch=args.bench_batch,
            seq_len=args.bench_seq,
            n_cols=args.bench_cols,
            block_n=args.block_n,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
