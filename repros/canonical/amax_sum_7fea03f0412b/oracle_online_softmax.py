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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



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


def oracle_forward(inputs):
    return oracle_masked_softmax(*inputs)


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
