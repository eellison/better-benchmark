"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DistillGPT2 layernorm-backward/dropout return tuple, including the two hidden-column reductions, position-embedding scatter-add, and token-embedding scatter-add added to `mm[:50257]`, whereas Inductor currently materializes the rowwise layernorm/dropout producer and schedules the sibling reductions and duplicate-index `index_put(accumulate=True)` outputs as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured embedding scatter-reduce template that keeps row-local layernorm reduction scalars live while feeding multiple column reductions and indexed accumulator destinations; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that emits the rowwise layernorm math once, accumulates the hidden-column reductions, and atomically scatters position and token gradients directly into their destination rows."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_sum_c01ffea2a781"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_distillgpt2_train_003_b6cfacd8"

BATCH = 32
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
POSITION_ROWS = 1024
VOCAB = 50257
BLOCK_SUM_H = 16



if triton is not None:

    @triton.jit
    def _seq_layernorm_scatter_kernel(
        mm48_ptr,
        gamma_ptr,
        saved_ptr,
        position_base_ptr,
        dropout_mask_ptr,
        mean_ptr,
        invstd_ptr,
        add_ptr,
        position_idx_ptr,
        token_idx_ptr,
        partial_sum_xhat_ptr,
        partial_sum_plain_ptr,
        position_out_ptr,
        vocab_out_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        BATCH_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        VOCAB_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        h_mask = h < HIDDEN_
        seq_offsets = seq * HIDDEN_ + h

        gamma = tl.load(gamma_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
        position_base = tl.load(position_base_ptr + seq_offsets, mask=h_mask, other=0.0).to(tl.float32)

        acc_sum_xhat = tl.zeros([BLOCK_H_], dtype=tl.float32)
        acc_sum_plain = tl.zeros([BLOCK_H_], dtype=tl.float32)
        acc_position = tl.zeros([BLOCK_H_], dtype=tl.float32)

        for b in tl.static_range(0, BATCH_):
            row = b * SEQ_ + seq
            offsets = row * HIDDEN_ + h

            upstream = tl.load(mm48_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            weighted = upstream * gamma
            keep = tl.load(dropout_mask_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
            saved_value = tl.load(saved_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + row).to(tl.float32)
            invstd = tl.load(invstd_ptr + row).to(tl.float32)
            normalized = (keep * (saved_value + position_base) - mean) * invstd

            row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(h_mask, weighted * normalized, 0.0), axis=0)
            layernorm_grad = invstd * (weighted - (row_sum + normalized * row_dot) / HIDDEN_)

            residual_grad = tl.load(add_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            embed_grad = (residual_grad + layernorm_grad) * keep

            acc_sum_xhat += tl.where(h_mask, upstream * normalized, 0.0)
            acc_sum_plain += tl.where(h_mask, upstream, 0.0)
            acc_position += tl.where(h_mask, embed_grad, 0.0)

            token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
            token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_, token_raw)
            token_value = tl.where(token_raw == -1, 0.0, embed_grad)
            token_valid = h_mask & (token_wrapped >= 0) & (token_wrapped < VOCAB_)
            tl.atomic_add(
                vocab_out_ptr + token_wrapped * HIDDEN_ + h,
                token_value,
                sem="relaxed",
                mask=token_valid,
            )

        position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
        position_wrapped = tl.where(position_raw < 0, position_raw + POSITION_ROWS_, position_raw)
        position_value = tl.where(position_raw == -1, 0.0, acc_position)
        position_valid = h_mask & (position_wrapped >= 0) & (position_wrapped < POSITION_ROWS_)
        tl.atomic_add(
            position_out_ptr + position_wrapped * HIDDEN_ + h,
            position_value,
            sem="relaxed",
            mask=position_valid,
        )

        partial_offsets = seq * HIDDEN_ + h
        tl.store(partial_sum_xhat_ptr + partial_offsets, acc_sum_xhat, mask=h_mask)
        tl.store(partial_sum_plain_ptr + partial_offsets, acc_sum_plain, mask=h_mask)

    @triton.jit
    def _finalize_hidden_sums_kernel(
        partial_sum_xhat_ptr,
        partial_sum_plain_ptr,
        out_sum_xhat_ptr,
        out_sum_plain_ptr,
        SEQ_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        h = tl.program_id(0) * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        n = tl.arange(0, BLOCK_N_)
        mask = (n[:, None] < SEQ_) & (h[None, :] < HIDDEN_)
        offsets = n[:, None] * HIDDEN_ + h[None, :]

        sum_xhat = tl.load(partial_sum_xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_plain = tl.load(partial_sum_plain_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out_mask = h < HIDDEN_
        tl.store(out_sum_xhat_ptr + h, tl.sum(sum_xhat, axis=0), mask=out_mask)
        tl.store(out_sum_plain_ptr + h, tl.sum(sum_plain, axis=0), mask=out_mask)


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def oracle_distillgpt2_embedding_scatter_reduce(
    mm: torch.Tensor,
    mm_48: torch.Tensor,
    arg2_1: torch.Tensor,
    arg39_1: torch.Tensor,
    arg41_1: torch.Tensor,
    arg42_1: torch.Tensor,
    arg43_1: torch.Tensor,
    arg44_1: torch.Tensor,
    add_35: torch.Tensor,
    arg40_1: torch.Tensor,
    arg0_1: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter: {_shape_param_0}")

    mm = mm.contiguous()
    mm_48 = mm_48.contiguous()
    arg2_1 = arg2_1.contiguous()
    arg39_1 = arg39_1.contiguous()
    arg41_1 = arg41_1.contiguous()
    arg42_1 = arg42_1.contiguous()
    arg43_1 = arg43_1.contiguous()
    arg44_1 = arg44_1.contiguous()
    add_35 = add_35.contiguous()
    arg40_1 = arg40_1.contiguous()
    arg0_1 = arg0_1.contiguous()

    assert mm.shape == (VOCAB + 3, HIDDEN)
    assert mm_48.shape == (ROWS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg39_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg41_1.shape == (1, SEQ, HIDDEN)
    assert arg42_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg43_1.shape == (BATCH, SEQ, 1)
    assert arg44_1.shape == (BATCH, SEQ, 1)
    assert add_35.shape == (BATCH, SEQ, HIDDEN)
    assert arg40_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)

    device = mm.device
    partial_sum_xhat = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_plain = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out_sum_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_plain = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = mm[:VOCAB].clone()
    position_out.zero_()

    _seq_layernorm_scatter_kernel[(SEQ,)](
        mm_48,
        arg2_1,
        arg39_1,
        arg41_1,
        arg42_1,
        arg43_1,
        arg44_1,
        add_35,
        arg40_1,
        arg0_1,
        partial_sum_xhat,
        partial_sum_plain,
        position_out,
        vocab_out,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BATCH_=BATCH,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_=VOCAB,
        BLOCK_H_=triton.next_power_of_2(HIDDEN),
        num_warps=8,
    )
    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, BLOCK_SUM_H),)](
        partial_sum_xhat,
        partial_sum_plain,
        out_sum_xhat,
        out_sum_plain,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_N_=triton.next_power_of_2(SEQ),
        BLOCK_H_=BLOCK_SUM_H,
        num_warps=8,
    )

    return out_sum_xhat, out_sum_plain, position_out, vocab_out


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
    return oracle_distillgpt2_embedding_scatter_reduce(*inputs)


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
