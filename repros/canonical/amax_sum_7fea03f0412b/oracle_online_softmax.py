"""
Oracle for amax_sum_7fea03f0412b: BERT attention masked softmax.

Repro pattern:
  arg0: i64[128, 128] attention key mask
  bmm:  f32[1536, 128, 128] attention scores

  valid = arg0 > 0
  scores = view(bmm, [128, 12, 128, 128]) / 8
  masked_scores = where(valid[:, None, None, :], scores, -1e9)
  softmax = exp(masked_scores - amax(masked_scores, -1)) / sum(...)
  dropout = (inductor_random > 0.1) * softmax * 1.1111111111111112
  output = view(dropout, [1536, 128, 128]).permute(0, 2, 1)

The full repro contains stochastic dropout.  The correctness check below is
explicitly for the deterministic reduction/mask core, with an additional check
that uses a supplied deterministic dropout mask.  It does not claim bit-exact
agreement with Inductor's random stream.

Since K=128 fits in one Triton block, this is a persistent row softmax: one
program handles one [K] row, keeps the row max and denominator as scalar
accumulators, and writes the normalized row without intermediate tensors.

Gap diagnosis (classification: NEW_PATTERN): this oracle lowers the masked
attention softmax, optional supplied dropout mask, and output layout view into
one persistent row kernel with scalar max/denominator accumulators and no
materialized amax, exp, sum, or div intermediates, while Inductor currently sees
the decomposed graph as separate masking, amax, exp/sum, normalization, RNG
dropout, and layout ops. Inductor cannot do this today because its generic
reduction scheduling does not recognize this masked softmax/dropout idiom as a
single online-softmax template, especially with the key mask built through
unsqueeze/repeat/view and the stochastic dropout boundary. The fix is to add an
Inductor pattern/template that canonicalizes masked last-dim attention softmax
with optional dropout into a persistent online softmax kernel using scalar
accumulators and fused epilogue/layout handling.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_7fea03f0412b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
N_HEADS = 12
Q_LEN = 128
K_LEN = 128
M_ROWS = BATCH * N_HEADS * Q_LEN
SCALE = 0.125
MASK_VALUE = -1000000000.0
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
def _persistent_masked_softmax_kernel(
    mask_ptr,
    bmm_ptr,
    dropout_mask_ptr,
    out_base_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    row = tl.program_id(0)
    bh = row // Q
    batch = bh // H

    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K

    bmm_vals = tl.load(
        bmm_ptr + row * K + cols,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    key_valid = tl.load(
        mask_ptr + batch * K + cols,
        mask=col_mask,
        other=0,
    ) > 0

    scaled = bmm_vals * 0.125
    scores = tl.where(col_mask, tl.where(key_valid, scaled, -1000000000.0), float("-inf"))

    row_max = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - row_max)
    denom = tl.sum(exp_scores, axis=0)
    out = exp_scores / denom

    if APPLY_DROPOUT:
        keep = tl.load(
            dropout_mask_ptr + row * K + cols,
            mask=col_mask,
            other=0,
        )
        out = tl.where(keep, out * DROPOUT_SCALE_CONST, 0.0)

    tl.store(out_base_ptr + row * K + cols, out, mask=col_mask)


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


def _make_check_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    mask = torch.randint(-2, 3, (BATCH, K_LEN), dtype=torch.int64, device="cuda")
    mask[0, :] = 1
    mask[1, :] = 0
    mask[2, ::2] = 1
    mask[2, 1::2] = -1

    bmm = torch.randn(
        BATCH * N_HEADS,
        Q_LEN,
        K_LEN,
        dtype=torch.float32,
        device="cuda",
    )
    dropout_mask = (
        torch.rand(BATCH, N_HEADS, Q_LEN, K_LEN, device="cuda") > DROPOUT_P
    )
    return mask, bmm, dropout_mask


def _reference(
    mask: torch.Tensor,
    bmm: torch.Tensor,
    dropout_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = bmm.view(BATCH, N_HEADS, Q_LEN, K_LEN) * SCALE
    key_valid = mask > 0
    masked = torch.where(
        key_valid[:, None, None, :],
        scores,
        torch.full((), MASK_VALUE, dtype=scores.dtype, device=scores.device),
    )
    out = torch.softmax(masked, dim=-1)
    if dropout_mask is not None:
        out = torch.where(dropout_mask, out * DROPOUT_SCALE, torch.zeros_like(out))
    return out.view(BATCH * N_HEADS, Q_LEN, K_LEN).permute(0, 2, 1)


def _launch_oracle(
    mask: torch.Tensor,
    bmm: torch.Tensor,
    out_base: torch.Tensor,
    dropout_mask: torch.Tensor | None = None,
    block_k: int = K_LEN,
) -> torch.Tensor:
    assert mask.shape == (BATCH, K_LEN), mask.shape
    assert mask.dtype == torch.int64, mask.dtype
    assert bmm.shape == (BATCH * N_HEADS, Q_LEN, K_LEN), bmm.shape
    assert bmm.dtype == torch.float32, bmm.dtype
    assert out_base.shape == bmm.shape, out_base.shape
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    apply_dropout = dropout_mask is not None
    if apply_dropout:
        assert dropout_mask.shape == (BATCH, N_HEADS, Q_LEN, K_LEN), dropout_mask.shape
        assert dropout_mask.dtype == torch.bool, dropout_mask.dtype
        dropout_flat = dropout_mask.reshape(M_ROWS, K_LEN)
    else:
        dropout_flat = out_base

    _persistent_masked_softmax_kernel[(M_ROWS,)](
        mask,
        bmm,
        dropout_flat,
        out_base,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        BLOCK_K=block_k,
        APPLY_DROPOUT=apply_dropout,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
    )
    return out_base.permute(0, 2, 1)


def oracle_masked_softmax(
    mask: torch.Tensor,
    bmm: torch.Tensor,
    dropout_mask: torch.Tensor | None = None,
    block_k: int = K_LEN,
) -> torch.Tensor:
    out_base = torch.empty_like(bmm)
    return _launch_oracle(mask, bmm, out_base, dropout_mask, block_k)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual - expected).abs()
    rel = diff / expected.abs().clamp_min(1e-8)
    return diff.max().item(), rel.max().item()


def run_check() -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    mask, bmm, dropout_mask = _make_check_inputs()

    with torch.no_grad():
        ref_core = _reference(mask, bmm)
        got_core = oracle_masked_softmax(mask, bmm)
        ref_dropout = _reference(mask, bmm, dropout_mask)
        got_dropout = oracle_masked_softmax(mask, bmm, dropout_mask)

    core_abs, core_rel = _max_diff(got_core, ref_core)
    dropout_abs, dropout_rel = _max_diff(got_dropout, ref_dropout)
    core_ok = torch.allclose(got_core, ref_core, rtol=1e-5, atol=1e-6)
    dropout_ok = torch.allclose(got_dropout, ref_dropout, rtol=1e-5, atol=1e-6)

    print(f"check deterministic mask+softmax: max_abs={core_abs:.6e} "
          f"max_rel={core_rel:.6e} allclose={core_ok}")
    print(f"check supplied dropout mask:      max_abs={dropout_abs:.6e} "
          f"max_rel={dropout_rel:.6e} allclose={dropout_ok}")
    print("note: stochastic inductor_random is intentionally not checked bit-exactly")
    print(f"Correctness: {'PASS' if core_ok and dropout_ok else 'FAIL'}")
    return core_ok and dropout_ok


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


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    mask, bmm, dropout_mask = _make_check_inputs()
    out_base = torch.empty_like(bmm)
    out_dropout_base = torch.empty_like(bmm)

    dense_bytes = M_ROWS * K_LEN * 4 * 2
    logical_mask_bytes = M_ROWS * K_LEN * 8
    dropout_bytes = M_ROWS * K_LEN

    print(f"oracle shape: bmm=f32[{BATCH * N_HEADS}, {Q_LEN}, {K_LEN}], "
          f"mask=i64[{BATCH}, {K_LEN}]")
    print(f"dense read+write traffic: {dense_bytes / 1e6:.1f} MB")
    print(f"logical mask reads if uncached: {logical_mask_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(mask, bmm, out_base),
            warmup=warmup,
            rep=rep,
        )
        dropout_us = _bench_cuda(
            lambda: _launch_oracle(mask, bmm, out_dropout_base, dropout_mask),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = dense_bytes / (oracle_us * 1e-6) / 1e12
    dropout_bw = (dense_bytes + dropout_bytes) / (dropout_us * 1e-6) / 1e12
    print(f"oracle deterministic mask+softmax: {oracle_us:.3f} us "
          f"({oracle_bw:.3f} TB/s dense bytes)")
    print(f"oracle with supplied dropout mask: {dropout_us:.3f} us "
          f"({dropout_bw:.3f} TB/s dense+dropout bytes)")

    if no_compile:
        return

    module = _load_repro_module()
    inputs = _cuda_inputs_from_repro()
    print("torch.compile full repro timings include stochastic dropout and final view")

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
    parser.add_argument("--rep", type=int, default=20, help="benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check():
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
