"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full GPT2 layernorm-backward and embedding-gradient tuple by fusing the hidden reductions, dropout/mask epilogue, and both duplicate-index scatter-add outputs into Triton row and scatter kernels, whereas Inductor materializes the `[4,512,768]` gradient producer and lowers the two `index_put(accumulate=True)` embedding updates plus sibling reductions as generic scheduled kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that shares one rowwise layernorm-backward producer across multiple indexed accumulation outputs and column reductions; the fix is SCATTER_REDUCE: add an Inductor lowering for embedding-backward scatter-reduce fed by rowwise reductions that emits shared reduction/scatter epilogues without materializing the intermediate."""
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



REPRO_ID = "sum_sum_sum_ae9b17407681"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
POSITION_ROWS = 1024
VOCAB_ROWS = 50257
PARTIAL_ROWS = SEQ
DROPOUT_SCALE = 1.1111111111111112



def _seq_layernorm_scatter_kernel(
    mm96_ptr,
    weight_ptr,
    xhat_ptr,
    row_scale_ptr,
    add71_ptr,
    dropout_mask_ptr,
    position_idx_ptr,
    token_idx_ptr,
    partial_sum_xxhat_ptr,
    partial_sum_x_ptr,
    position_out_ptr,
    vocab_out_ptr,
    BATCH_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    h_mask = h < HIDDEN_
    weight = tl.load(weight_ptr + h, mask=h_mask, other=0.0).to(tl.float32)

    acc_sum_xxhat = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_sum_x = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_position_grad = tl.zeros([BLOCK_H], dtype=tl.float32)

    for b in tl.static_range(0, BATCH_):
        row = b * SEQ_ + seq
        offsets = row * HIDDEN_ + h

        x = tl.load(mm96_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
        weighted = x * weight
        row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(h_mask, weighted * xhat, 0.0), axis=0)
        scale = tl.load(row_scale_ptr + row).to(tl.float32)
        ln_grad = scale * (weighted * HIDDEN_ - row_sum - xhat * row_dot)

        add_value = tl.load(add71_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
        keep = tl.load(dropout_mask_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
        grad = (add_value + ln_grad) * keep * DROPOUT_SCALE_

        acc_sum_xxhat += tl.where(h_mask, x * xhat, 0.0)
        acc_sum_x += tl.where(h_mask, x, 0.0)
        acc_position_grad += tl.where(h_mask, grad, 0.0)

        token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
        token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
        token_valid = (
            h_mask
            & (token_raw != -1)
            & (token_wrapped >= 0)
            & (token_wrapped < VOCAB_ROWS_)
        )
        tl.atomic_add(
            vocab_out_ptr + token_wrapped * HIDDEN_ + h,
            grad,
            sem="relaxed",
            mask=token_valid,
        )

    position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
    position_wrapped = tl.where(position_raw < 0, position_raw + POSITION_ROWS_, position_raw)
    position_valid = (
        h_mask
        & (position_raw != -1)
        & (position_wrapped >= 0)
        & (position_wrapped < POSITION_ROWS_)
    )
    tl.atomic_add(
        position_out_ptr + position_wrapped * HIDDEN_ + h,
        acc_position_grad,
        sem="relaxed",
        mask=position_valid,
    )

    partial_offsets = seq * HIDDEN_ + h
    tl.store(partial_sum_xxhat_ptr + partial_offsets, acc_sum_xxhat, mask=h_mask)
    tl.store(partial_sum_x_ptr + partial_offsets, acc_sum_x, mask=h_mask)


@triton.jit
def _finalize_hidden_sums_kernel(
    partial_sum_xxhat_ptr,
    partial_sum_x_ptr,
    out_sum_xxhat_ptr,
    out_sum_x_ptr,
    ROW_BLOCKS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    n = tl.arange(0, BLOCK_N)
    mask = (n[:, None] < ROW_BLOCKS_) & (h[None, :] < HIDDEN_)
    offsets = n[:, None] * HIDDEN_ + h[None, :]

    sum_xxhat = tl.load(partial_sum_xxhat_ptr + offsets, mask=mask, other=0.0)
    sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_sum_xxhat_ptr + h, tl.sum(sum_xxhat, axis=0), mask=h < HIDDEN_)
    tl.store(out_sum_x_ptr + h, tl.sum(sum_x, axis=0), mask=h < HIDDEN_)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_96,
        arg2_1,
        arg77_1,
        arg284_1,
        add_71,
        arg76_1,
        arg75_1,
        arg0_1,
        mm,
        _shape_param_0,
    ) = inputs
    del _shape_param_0
    return (
        mm_96.contiguous(),
        arg2_1.contiguous(),
        arg77_1.contiguous(),
        arg284_1.contiguous(),
        add_71.contiguous(),
        arg76_1.contiguous(),
        arg75_1.contiguous(),
        arg0_1.contiguous(),
        mm.contiguous(),
    )


def oracle_structured_scatter_reduce(
    mm_96: torch.Tensor,
    arg2_1: torch.Tensor,
    arg77_1: torch.Tensor,
    arg284_1: torch.Tensor,
    add_71: torch.Tensor,
    arg76_1: torch.Tensor,
    arg75_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_96.shape == (ROWS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg77_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg284_1.shape == (BATCH, SEQ, 1)
    assert add_71.shape == (BATCH, SEQ, HIDDEN)
    assert arg76_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg75_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm.shape == (VOCAB_ROWS, HIDDEN)

    device = mm_96.device
    partial_sum_xxhat = torch.empty((PARTIAL_ROWS, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((PARTIAL_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_sum_xxhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = mm.clone()
    position_out.zero_()

    _seq_layernorm_scatter_kernel[(SEQ,)](
        mm_96,
        arg2_1,
        arg77_1,
        arg284_1,
        add_71,
        arg76_1,
        arg75_1,
        arg0_1,
        partial_sum_xxhat,
        partial_sum_x,
        position_out,
        vocab_out,
        BATCH_=BATCH,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_H=triton.next_power_of_2(HIDDEN),
        DROPOUT_SCALE_=DROPOUT_SCALE,
        num_warps=8,
    )

    finalize_block_h = 16
    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, finalize_block_h),)](
        partial_sum_xxhat,
        partial_sum_x,
        out_sum_xxhat,
        out_sum_x,
        ROW_BLOCKS_=PARTIAL_ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_N=triton.next_power_of_2(PARTIAL_ROWS),
        BLOCK_H=finalize_block_h,
        num_warps=8,
    )

    return out_sum_xxhat, out_sum_x, position_out, vocab_out


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def oracle_forward(inputs):
    return oracle_structured_scatter_reduce(*inputs)


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
