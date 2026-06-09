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


@oracle_impl(hardware="H100", shapes="(T([128, 50400], f32), T([1, 128], i64), T([], f32), S([1, 128, 50400]), S([-1, 50400]))")
def oracle_forward(inputs):
    return oracle_shifted_ignore_index_cross_entropy_mean(*inputs)


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
