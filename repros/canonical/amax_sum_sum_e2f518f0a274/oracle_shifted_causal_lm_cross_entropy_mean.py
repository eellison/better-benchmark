"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-2 shifted-label ignore-index cross-entropy mean returned by Repro.forward, including the constant-pad/slice/clone label shift, logits view to [2048, 50257], stable row logsumexp, safe masked target gather, zero-loss ignore guard, valid-count reduction, loss reduction, and final scalar division in Triton kernels, whereas Inductor currently lowers the decomposed constant_pad_nd/slice/clone/view/amax/sub/exp/sum/log/sub/ne/where/gather/squeeze/neg/where/sum/count/div graph as separate generic reductions, pointwise kernels, gather work, and scalar reductions that materialize the full log-softmax intermediate; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize shifted-label ignore-index cross entropy with a gathered target loss and sibling valid-count reduction into a fused online row-reduction plus scalar epilogue; the fix is NEW_PATTERN: add an Inductor lowering for shifted log_softmax plus masked gather plus loss/count mean that emits the online cross-entropy row kernel and small final reduction directly."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "amax_sum_sum_e2f518f0a274"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ_LEN = 512
N_ROWS = BATCH * SEQ_LEN
N_COLS = 50257


if triton is not None:

    @triton.jit
    def _shifted_xent_rows_kernel(
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
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
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


def make_inputs(seed: int, inject_ignores: bool) -> tuple[Any, ...]:
    module = _load_repro_module()
    torch.manual_seed(seed)
    inputs = list(module.make_inputs())
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
    if inject_ignores:
        tokens = inputs[0].clone()
        flat = tokens.view(-1)
        flat[17::257] = -100
        tokens[:, 1] = torch.clamp(tokens[:, 1], min=0)
        inputs[0] = tokens
    return tuple(inputs)


def _launch_oracle(
    tokens: torch.Tensor,
    logits_2d: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert tokens.is_cuda and logits_2d.is_cuda
    assert tokens.dtype == torch.int64 and tokens.shape == (BATCH, SEQ_LEN)
    assert logits_2d.dtype == torch.float32 and logits_2d.shape == (N_ROWS, N_COLS)
    assert loss_per_row.shape == (N_ROWS,) and loss_per_row.dtype == torch.float32
    assert valid_per_row.shape == (N_ROWS,) and valid_per_row.dtype == torch.float32
    assert out.shape == () and out.dtype == torch.float32

    _shifted_xent_rows_kernel[(N_ROWS,)](
        tokens,
        logits_2d,
        loss_per_row,
        valid_per_row,
        n_cols=N_COLS,
        seq_len=SEQ_LEN,
        block_n=_effective_block_n(block_n, N_COLS),
        num_warps=num_warps,
    )

    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        out,
        n_rows=N_ROWS,
        block_m=triton.next_power_of_2(N_ROWS),
        num_warps=8,
    )
    return out


def oracle_shifted_causal_lm_cross_entropy_mean(
    arg1_1: torch.Tensor,
    arg0_1: torch.Tensor,
    _shape_param_0,
    *,
    block_n: int = 4096,
    num_warps: int = 8,
) -> torch.Tensor:
    if tuple(_shape_param_0) != (-1, N_COLS):
        raise ValueError(f"unexpected shape parameter: {_shape_param_0}")
    logits_2d = arg0_1.view(_shape_param_0)
    loss_per_row = torch.empty((N_ROWS,), device=arg0_1.device, dtype=torch.float32)
    valid_per_row = torch.empty((N_ROWS,), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty((), device=arg0_1.device, dtype=torch.float32)
    return _launch_oracle(
        arg1_1,
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
    return (value,)


def _compare_outputs(ref: Any, got: Any, rtol: float, atol: float) -> bool:
    ref_items = _as_tuple(ref)
    got_items = _as_tuple(got)
    if len(ref_items) != len(got_items):
        print(f"output arity mismatch: ref={len(ref_items)} oracle={len(got_items)}")
        return False

    ok = True
    for idx, (ref_item, got_item) in enumerate(zip(ref_items, got_items)):
        if not isinstance(ref_item, torch.Tensor) or not isinstance(got_item, torch.Tensor):
            item_ok = ref_item == got_item
            print(f"output[{idx}]: ref={ref_item!r} oracle={got_item!r} equal={item_ok}")
            ok = ok and bool(item_ok)
            continue

        metadata_ok = ref_item.shape == got_item.shape and ref_item.dtype == got_item.dtype
        ref_f = ref_item.float()
        got_f = got_item.float()
        diff = (ref_f - got_f).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        rel = diff / ref_f.abs().clamp_min(1e-8)
        max_rel = rel.max().item() if rel.numel() else 0.0
        values_ok = torch.allclose(ref_f, got_f, rtol=rtol, atol=atol, equal_nan=True)
        item_ok = metadata_ok and values_ok
        print(
            f"output[{idx}]: shape={list(ref_item.shape)} dtype={ref_item.dtype} "
            f"ref={ref_item.detach().flatten()[:3].tolist()} "
            f"oracle={got_item.detach().flatten()[:3].tolist()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"metadata={metadata_ok} allclose={values_ok}"
        )
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
    inputs = make_inputs(seed=args.seed, inject_ignores=True)
    model = module.Repro().cuda()

    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_shifted_causal_lm_cross_entropy_mean(
            *inputs,
            block_n=args.block_n,
            num_warps=args.num_warps,
        )
        torch.cuda.synchronize()

    tokens = inputs[0]
    shifted = torch.empty_like(tokens)
    shifted[:, :-1] = tokens[:, 1:]
    shifted[:, -1] = -100
    valid_count = (shifted != -100).sum().item()
    print(
        f"check full Repro.forward shifted xent mean: tokens=i64[{BATCH}, {SEQ_LEN}] "
        f"logits=f32[{N_ROWS}, {N_COLS}] valid_count={valid_count}"
    )
    ok = _compare_outputs(ref, got, rtol=args.rtol, atol=args.atol)
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(args: argparse.Namespace) -> None:
    _require_cuda()
    inputs = make_inputs(seed=args.seed, inject_ignores=True)
    tokens, logits, shape_2d = inputs
    logits_2d = logits.view(shape_2d)
    loss_per_row = torch.empty((N_ROWS,), device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty((N_ROWS,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)

    read_bytes = N_ROWS * N_COLS * 4 + N_ROWS * (8 + 4)
    write_bytes = N_ROWS * 8 + 4
    total_bytes = read_bytes + write_bytes
    print(
        f"bench full-scope shifted xent mean: tokens=i64[{BATCH}, {SEQ_LEN}] "
        f"logits=f32[{N_ROWS}, {N_COLS}]"
    )
    print(f"logical oracle traffic={total_bytes / 1e6:.3f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                tokens,
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block-n", type=int, default=4096)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("choose --check and/or --bench")

    if args.check and not run_check(args):
        sys.exit(1)
    if args.bench:
        run_bench(args)


if __name__ == "__main__":
    with torch.no_grad():
        main()
