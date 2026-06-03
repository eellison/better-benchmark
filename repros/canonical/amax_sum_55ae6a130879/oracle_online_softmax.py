"""
Gap diagnosis (classification: NEW_PATTERN): this oracle lowers the DeBERTa attention softmax, optional supplied dropout mask, and final output layout view into one persistent last-dimension Triton kernel that keeps the row max and denominator in scalar/vector registers and writes the normalized row directly, while Inductor currently lowers the decomposed amax/sub/exp/sum/div/dropout/permute graph as generic reduction and pointwise/layout work with separate materialization boundaries around stochastic dropout. Inductor cannot do this today because the scheduler does not canonicalize this unmasked attention-softmax-with-dropout idiom into a single online/persistent softmax template once the graph includes RNG and the trailing layout-only permute. The fix is a NEW_PATTERN lowering that recognizes last-dim attention softmax with optional dropout epilogue and emits a persistent online softmax kernel with fused epilogue/layout handling.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_55ae6a130879"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
N_HEADS = 24
Q_LEN = 512
K_LEN = 512
BH = BATCH * N_HEADS
M_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)

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
def _persistent_softmax_dropout_kernel(
    bmm_ptr,
    dropout_mask_ptr,
    out_base_ptr,
    n_cols: tl.constexpr,
    block_n: tl.constexpr,
    apply_dropout: tl.constexpr,
    dropout_scale: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_n)
    col_mask = cols < n_cols

    x = tl.load(
        bmm_ptr + row * n_cols + cols,
        mask=col_mask,
        other=-float("inf"),
    ).to(tl.float32)

    row_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - row_max)
    denom = tl.sum(exp_x, axis=0)
    out = exp_x / denom

    if apply_dropout:
        keep = tl.load(
            dropout_mask_ptr + row * n_cols + cols,
            mask=col_mask,
            other=0,
        )
        out = tl.where(keep, out * dropout_scale, 0.0)

    tl.store(out_base_ptr + row * n_cols + cols, out, mask=col_mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cuda_inputs_from_repro() -> tuple:
    module = _load_repro_module()
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)


def _make_inputs(
    bh: int,
    q_len: int,
    k_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    bmm = torch.randn(
        (bh, q_len, k_len),
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
    dropout_mask = torch.rand(
        (bh, q_len, k_len),
        device="cuda",
        generator=gen,
    ) > DROPOUT_P
    return bmm, dropout_mask


def eager_reference(
    bmm: torch.Tensor,
    dropout_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    softmax = torch.softmax(bmm, dim=-1)
    if dropout_mask is not None:
        softmax = torch.where(
            dropout_mask,
            softmax * DROPOUT_SCALE,
            torch.zeros_like(softmax),
        )
    return softmax.permute(0, 2, 1)


def _launch_oracle(
    bmm: torch.Tensor,
    out_base: torch.Tensor,
    dropout_mask: torch.Tensor | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    assert bmm.is_cuda and out_base.is_cuda
    assert bmm.dtype == torch.float32 and out_base.dtype == torch.float32
    assert bmm.ndim == 3 and out_base.shape == bmm.shape
    assert bmm.is_contiguous() and out_base.is_contiguous()

    bh, q_len, k_len = bmm.shape
    n_rows = bh * q_len
    block = block_n if block_n is not None else triton.next_power_of_2(k_len)
    assert block >= k_len and (block & (block - 1)) == 0

    apply_dropout = dropout_mask is not None
    if apply_dropout:
        assert dropout_mask.shape == bmm.shape
        assert dropout_mask.dtype == torch.bool and dropout_mask.is_contiguous()
        dropout_ptr = dropout_mask
    else:
        dropout_ptr = out_base

    _persistent_softmax_dropout_kernel[(n_rows,)](
        bmm,
        dropout_ptr,
        out_base,
        n_cols=k_len,
        block_n=block,
        apply_dropout=apply_dropout,
        dropout_scale=DROPOUT_SCALE,
    )
    return out_base.permute(0, 2, 1)


def oracle_online_softmax(
    bmm: torch.Tensor,
    dropout_mask: torch.Tensor | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    out_base = torch.empty_like(bmm)
    return _launch_oracle(bmm, out_base, dropout_mask=dropout_mask, block_n=block_n)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual - expected).abs()
    rel = diff / expected.abs().clamp_min(1e-8)
    return diff.max().item(), rel.max().item()


def run_check(
    bh: int,
    q_len: int,
    k_len: int,
    block_n: int | None,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    bmm, dropout_mask = _make_inputs(bh, q_len, k_len, seed=1234)
    with torch.no_grad():
        ref_core = eager_reference(bmm)
        got_core = oracle_online_softmax(bmm, block_n=block_n)
        ref_dropout = eager_reference(bmm, dropout_mask)
        got_dropout = oracle_online_softmax(
            bmm,
            dropout_mask=dropout_mask,
            block_n=block_n,
        )
        torch.cuda.synchronize()

    core_abs, core_rel = _max_diff(got_core, ref_core)
    dropout_abs, dropout_rel = _max_diff(got_dropout, ref_dropout)
    core_ok = torch.allclose(got_core, ref_core, rtol=1e-5, atol=1e-6)
    dropout_ok = torch.allclose(got_dropout, ref_dropout, rtol=1e-5, atol=1e-6)

    print(
        f"check deterministic softmax: shape=({bh}, {q_len}, {k_len}) "
        f"max_abs={core_abs:.6e} max_rel={core_rel:.6e} allclose={core_ok}"
    )
    print(
        f"check supplied dropout mask: shape=({bh}, {q_len}, {k_len}) "
        f"max_abs={dropout_abs:.6e} max_rel={dropout_rel:.6e} allclose={dropout_ok}"
    )
    print("note: stochastic inductor_random is intentionally not checked bit-exactly")
    print(f"Correctness: {'PASS' if core_ok and dropout_ok else 'FAIL'}")
    return bool(core_ok and dropout_ok)


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


def run_bench(
    bh: int,
    q_len: int,
    k_len: int,
    block_n: int | None,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    bmm, dropout_mask = _make_inputs(bh, q_len, k_len, seed=4321)
    out_base = torch.empty_like(bmm)
    out_dropout_base = torch.empty_like(bmm)

    dense_bytes = bh * q_len * k_len * 4 * 2
    dropout_bytes = bh * q_len * k_len

    print(f"oracle shape: bmm=f32[{bh}, {q_len}, {k_len}]")
    print(f"dense read+write traffic: {dense_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                bmm,
                out_base,
                block_n=block_n,
            ),
            warmup=warmup,
            rep=rep,
        )
        dropout_us = _bench_cuda(
            lambda: _launch_oracle(
                bmm,
                out_dropout_base,
                dropout_mask=dropout_mask,
                block_n=block_n,
            ),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = dense_bytes / (oracle_us * 1e-6) / 1e12
    dropout_bw = (dense_bytes + dropout_bytes) / (dropout_us * 1e-6) / 1e12
    print(f"oracle deterministic softmax: {oracle_us:.3f} us ({oracle_bw:.3f} TB/s)")
    print(
        f"oracle with supplied dropout mask: {dropout_us:.3f} us "
        f"({dropout_bw:.3f} TB/s dense+dropout bytes)"
    )

    if no_compile:
        return
    if (bh, q_len, k_len) != (BH, Q_LEN, K_LEN):
        print("torch.compile full repro skipped for noncanonical benchmark shape")
        return

    module = _load_repro_module()
    inputs = _cuda_inputs_from_repro()
    print("torch.compile full repro timings include stochastic dropout and final permute")

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
    parser.add_argument("--warmup", type=int, default=5, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=20, help="benchmark repetitions")
    parser.add_argument("--block-n", type=int, default=K_LEN, help="Triton reduction tile size")
    parser.add_argument("--check-bh", type=int, default=4)
    parser.add_argument("--check-q", type=int, default=32)
    parser.add_argument("--check-k", type=int, default=K_LEN)
    parser.add_argument("--bench-bh", type=int, default=BH)
    parser.add_argument("--bench-q", type=int, default=Q_LEN)
    parser.add_argument("--bench-k", type=int, default=K_LEN)
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
        check_block = max(args.block_n, triton.next_power_of_2(args.check_k))
        ok = run_check(
            bh=args.check_bh,
            q_len=args.check_q,
            k_len=args.check_k,
            block_n=check_block,
        )
        if not ok:
            sys.exit(1)

    if args.bench:
        bench_block = max(args.block_n, triton.next_power_of_2(args.bench_k))
        run_bench(
            bh=args.bench_bh,
            q_len=args.bench_q,
            k_len=args.bench_k,
            block_n=bench_block,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
