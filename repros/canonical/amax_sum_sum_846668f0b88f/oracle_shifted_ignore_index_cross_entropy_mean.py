"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full shifted ignore-index cross-entropy mean returned by repro.py by shifting token labels, reading each logits row once with scalar online max and sum-exp accumulators plus one target-logit load, writing only per-row loss and valid-count summaries, and reducing those summaries to the scalar mean, whereas Inductor currently lowers the decomposed pad/slice/view/amax/sub/exp/sum/log/gather/where/sum/div graph as separate generic softmax and loss-reduction work that materializes and rereads the full log-softmax-sized intermediate; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize shifted-label ignore-index cross entropy into a fused online softmax row template with a valid-count side reduction; the fix is NEW_PATTERN: add an Inductor lowering for shifted log_softmax+gather+masked-mean cross entropy that emits the online row accumulator kernel and small final scalar reduction directly."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_sum_846668f0b88f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
SEQ_LEN = 128
N_COLS = 50400
N_ROWS = BATCH * SEQ_LEN


@triton.jit
def _shifted_xent_rows_kernel(
    tokens_ptr,
    logits_ptr,
    fill_ptr,
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
    fill = tl.load(fill_ptr).to(tl.float32)
    x_label = tl.load(
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

    loss = row_max + tl.log(denom) - x_label
    loss = tl.where(is_valid, loss, fill)

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


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _launch_oracle(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    fill: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert tokens.is_cuda and logits.is_cuda and fill.is_cuda
    assert tokens.dtype == torch.int64
    assert logits.dtype == torch.float32 and logits.ndim == 2
    assert fill.shape == () and fill.dtype == torch.float32
    assert tokens.ndim == 2

    n_rows, n_cols = logits.shape
    batch, seq_len = tokens.shape
    assert n_rows == batch * seq_len
    assert loss_per_row.shape == (n_rows,) and loss_per_row.dtype == torch.float32
    assert valid_per_row.shape == (n_rows,) and valid_per_row.dtype == torch.float32
    assert out.shape == () and out.dtype == torch.float32

    _shifted_xent_rows_kernel[(n_rows,)](
        tokens,
        logits,
        fill,
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


def oracle_shifted_ignore_index_cross_entropy_mean(
    addmm_56: torch.Tensor,
    arg314_1: torch.Tensor,
    full_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    block_n: int = 4096,
    num_warps: int = 8,
) -> torch.Tensor:
    n_cols = int(_shape_param_1[-1])
    logits = addmm_56.view(_shape_param_1)
    tokens = arg314_1
    n_rows = logits.shape[0]

    loss_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    return _launch_oracle(
        tokens,
        logits,
        full_1,
        loss_per_row,
        valid_per_row,
        out,
        block_n=min(block_n, triton.next_power_of_2(n_cols)),
        num_warps=num_warps,
    )


def _make_synthetic_inputs(
    batch: int,
    seq_len: int,
    n_cols: int,
    seed: int,
) -> tuple[Any, ...]:
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2 so the shifted target has a valid slot")

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    logits = torch.randn(
        (batch * seq_len, n_cols),
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
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
    tokens[:, 1] = torch.clamp(tokens[:, 1], min=0)
    fill = torch.randn((), device="cuda", dtype=torch.float32, generator=gen)
    return logits, tokens, fill, [batch, seq_len, n_cols], [-1, n_cols]


def _make_check_inputs(module, batch: int, seq_len: int, n_cols: int, seed: int) -> tuple[Any, ...]:
    if (batch, seq_len, n_cols) == (BATCH, SEQ_LEN, N_COLS):
        torch.manual_seed(seed)
        return tuple(module.make_inputs())
    return _make_synthetic_inputs(batch, seq_len, n_cols, seed)


def _compare_outputs(
    ref: Any,
    got: Any,
    rtol: float,
    atol: float,
) -> bool:
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

            diff = (ref_item - got_item).abs()
            max_abs = diff.max().item() if diff.numel() else 0.0
            denom = ref_item.abs().clamp_min(1e-8)
            max_rel = (diff / denom).max().item() if diff.numel() else 0.0
            item_ok = torch.allclose(ref_item, got_item, rtol=rtol, atol=atol, equal_nan=True)
            print(
                f"output[{idx}]: shape={tuple(ref_item.shape)} dtype={ref_item.dtype} "
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --check")

    module = _load_repro_module()
    inputs = _make_check_inputs(
        module,
        batch=args.check_batch,
        seq_len=args.check_seq,
        n_cols=args.check_cols,
        seed=args.seed,
    )
    model = module.Repro().cuda()

    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_shifted_ignore_index_cross_entropy_mean(
            *inputs,
            block_n=args.block_n,
            num_warps=args.num_warps,
        )
        torch.cuda.synchronize()

    ok = _compare_outputs(ref, got, rtol=args.rtol, atol=args.atol)
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --bench")

    inputs = _make_synthetic_inputs(args.bench_batch, args.bench_seq, args.bench_cols, args.seed)
    logits, tokens, fill, _shape_param_0, _shape_param_1 = inputs
    n_rows, n_cols = logits.shape
    loss_per_row = torch.empty((n_rows,), device="cuda", dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device="cuda", dtype=torch.float32)
    out = torch.empty((), device="cuda", dtype=torch.float32)
    block_n = min(args.block_n, triton.next_power_of_2(n_cols))

    read_bytes = n_rows * n_cols * 4 + n_rows * 8 + 4
    write_bytes = n_rows * 8 + 4
    total_bytes = read_bytes + write_bytes
    print(
        f"oracle shape: logits=f32[{n_rows}, {n_cols}], "
        f"tokens=i64[{tokens.shape[0]}, {tokens.shape[1]}], fill=f32[]"
    )
    print(f"single-pass logical traffic: {total_bytes / 1e6:.3f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                tokens,
                logits,
                fill,
                loss_per_row,
                valid_per_row,
                out,
                block_n=block_n,
                num_warps=args.num_warps,
            ),
            warmup=args.warmup,
            rep=args.rep,
        )
    oracle_bw = total_bytes / (oracle_us * 1e-6) / 1e12
    print(f"oracle shifted ignore-index xent mean: {oracle_us:.3f} us ({oracle_bw:.3f} TB/s)")

    if args.compile:
        import torch._dynamo as torch_dynamo

        module = _load_repro_module()
        model = module.Repro().cuda()
        torch_dynamo.reset()
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled_us = _bench_cuda(lambda: compiled(*inputs), warmup=args.warmup, rep=args.rep)
        print(f"torch.compile full repro: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the full repro.py output tuple against the oracle")
    parser.add_argument("--bench", action="store_true", help="benchmark the oracle implementation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block-n", type=int, default=4096)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=5e-5)
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--check-batch", type=int, default=BATCH)
    parser.add_argument("--check-seq", type=int, default=SEQ_LEN)
    parser.add_argument("--check-cols", type=int, default=N_COLS)
    parser.add_argument("--bench-batch", type=int, default=BATCH)
    parser.add_argument("--bench-seq", type=int, default=SEQ_LEN)
    parser.add_argument("--bench-cols", type=int, default=N_COLS)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--compile", action="store_true", help="also time torch.compile on the full repro")
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
