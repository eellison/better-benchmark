"""
Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full
ignore-index cross-entropy mean by folding the `arg665_1` bias add into a
single-pass online logsumexp over each `[2048, 8008]` row, eagerly loading the
target logit, emitting only per-row loss and valid-count scalars, and reducing
those to the final scalar division. Inductor currently lowers the decomposed
add/amax/sub/exp/sum/log/log-softmax/gather/mask/sum/count/div graph as generic
row reductions plus pointwise/gather work, so it materializes and rereads the
large log-softmax-sized intermediate instead of recognizing that the gathered
negative log-probability can be computed as `max + log(sum_exp) - x_target`.
The fix is a NEW_PATTERN lowering for biased log_softmax+gather+masked-mean
cross entropy that emits an online accumulator row template with the scalar
loss/count epilogue.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_sum_f9833ae619a1"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ_LEN = 128
N_ROWS = BATCH * SEQ_LEN
N_COLS = 8008

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
def _biased_online_xent_rows_kernel(
    labels_ptr,
    logits_ptr,
    bias_ptr,
    loss_ptr,
    valid_ptr,
    n_cols: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)

    row_start = row * n_cols
    x_label = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
    ).to(tl.float32)
    x_label += tl.load(bias_ptr + safe_label, mask=is_valid, other=0.0).to(tl.float32)

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
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + b

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


def _launch_oracle(
    arg0_1: torch.Tensor,
    mm: torch.Tensor,
    arg665_1: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert arg0_1.is_cuda and mm.is_cuda and arg665_1.is_cuda
    assert arg0_1.dtype == torch.int64 and arg0_1.numel() == mm.shape[0]
    assert mm.dtype == torch.float32 and mm.ndim == 2
    assert arg665_1.dtype == torch.float32 and arg665_1.numel() == mm.shape[1]
    assert loss_per_row.shape == (mm.shape[0],) and loss_per_row.dtype == torch.float32
    assert valid_per_row.shape == (mm.shape[0],) and valid_per_row.dtype == torch.float32
    assert out.shape == () and out.dtype == torch.float32

    n_rows, n_cols = mm.shape
    _biased_online_xent_rows_kernel[(n_rows,)](
        arg0_1,
        mm,
        arg665_1,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
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
    arg0_1: torch.Tensor,
    mm: torch.Tensor,
    arg665_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    *,
    block_n: int = 2048,
    num_warps: int = 8,
) -> torch.Tensor:
    n_rows = mm.shape[0]
    loss_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    out = torch.empty((), device=mm.device, dtype=torch.float32)
    return _launch_oracle(
        arg0_1,
        mm,
        arg665_1,
        loss_per_row,
        valid_per_row,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_inputs(module, seed: int, inject_ignores: bool) -> tuple:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = list(module.make_inputs())
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

    if inject_ignores:
        labels = inputs[0].clone()
        flat = labels.view(-1)
        flat[::257] = -100
        inputs[0] = labels
    return tuple(inputs)


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


def _compile_with_config(module, inputs: tuple, config: dict[str, object], warmup: int):
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


def run_check(block_n: int, num_warps: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234, inject_ignores=True)
    model = module.Repro().cuda()

    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_online_softmax_xent_mean(
            *inputs,
            block_n=block_n,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

    abs_diff = (got - ref).abs().item()
    rel_diff = (got - ref).abs().div(ref.abs().clamp_min(1e-8)).item()
    ok = torch.allclose(got, ref, rtol=5e-5, atol=5e-5)
    valid = (inputs[0] != -100).sum().item()

    print(
        "check biased online xent mean: "
        f"shape=labels[{BATCH}, {SEQ_LEN}], logits[{N_ROWS}, {N_COLS}], "
        f"valid={valid} ref={ref.item():.8f} oracle={got.item():.8f} "
        f"max_abs={abs_diff:.6e} max_rel={rel_diff:.6e} allclose={ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(block_n: int, num_warps: int, warmup: int, rep: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321, inject_ignores=True)
    labels, mm, bias = inputs[:3]
    n_rows, n_cols = mm.shape
    loss_per_row = torch.empty((n_rows,), device="cuda", dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device="cuda", dtype=torch.float32)
    out = torch.empty((), device="cuda", dtype=torch.float32)

    read_bytes = n_rows * n_cols * 4 + n_cols * 4 + n_rows * 8
    write_bytes = n_rows * 8 + 4
    total_bytes = read_bytes + write_bytes

    print(
        f"oracle shape: labels=i64[{BATCH}, {SEQ_LEN}], "
        f"mm=f32[{n_rows}, {n_cols}], bias=f32[1, {n_cols}]"
    )
    print(f"single-pass logical traffic: {total_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                labels,
                mm,
                bias,
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
    print(f"oracle biased online xent mean: {oracle_us:.3f} us ({oracle_bw:.3f} TB/s)")

    if no_compile:
        return

    print("torch.compile full repro timings include bias add, gather mask, count, and final div")
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module, inputs, config, warmup)
            us = _bench_cuda(lambda: compiled(*inputs), warmup=warmup, rep=rep)
            print(f"torch.compile {label}: {us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-n", type=int, default=2048, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=8, help="Triton warps for row kernel")
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
        ok = run_check(block_n=args.block_n, num_warps=args.num_warps)
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            block_n=args.block_n,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
