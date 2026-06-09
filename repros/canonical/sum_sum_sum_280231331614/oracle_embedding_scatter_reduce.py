"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GPT-2 large layernorm-backward plus dropout, position-embedding scatter-add, token-embedding scatter-add, and sibling hidden-column sums from one rowwise producer, whereas Inductor currently materializes the layernorm-gradient and dropout tensors then schedules the two sums and two index_put(accumulate=True) embedding-gradient consumers as separate generic reduction/scatter kernels; Inductor cannot do this today because scheduler/codegen has no structured embedding scatter-reduce template that shares rowwise layernorm reductions across sibling column reductions and multiple duplicate-index accumulation outputs with a dense add epilogue; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that computes each token row producer once, accumulates compatible hidden-column reductions, and atomically scatters position and token gradients directly into their destination rows."""
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
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None



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

REPRO_ID = "sum_sum_sum_280231331614"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_hf_gpt2_large_train_002_634af270"

BATCH = 4
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 1280
POSITION_ROWS = 1024
VOCAB = 50257
DROPOUT_SCALE = 1.1111111111111112
BLOCK_SUM_H = 16



if triton is not None:

    @triton.jit
    def _seq_layernorm_scatter_kernel(
        mm_ptr,
        gamma_ptr,
        xhat_ptr,
        inv_factor_ptr,
        add_ptr,
        dropout_mask_ptr,
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
        BLOCK_H: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)
        h_mask = h < HIDDEN_

        gamma = tl.load(gamma_ptr + h, mask=h_mask, other=0.0).to(tl.float32)

        acc_sum_xhat = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_sum_plain = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_position = tl.zeros([BLOCK_H], dtype=tl.float32)

        for b in tl.static_range(0, BATCH_):
            row = b * SEQ_ + seq
            offsets = row * HIDDEN_ + h
            mm = tl.load(mm_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            saved = tl.load(xhat_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            weighted = mm * gamma

            row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(h_mask, weighted * saved, 0.0), axis=0)
            inv_factor = tl.load(inv_factor_ptr + row).to(tl.float32)
            add = tl.load(add_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            keep = tl.load(dropout_mask_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
            grad = add + inv_factor * (weighted * HIDDEN_ - row_sum - saved * row_dot)
            grad = grad * keep * DROPOUT_SCALE_

            acc_sum_xhat += tl.where(h_mask, mm * saved, 0.0)
            acc_sum_plain += tl.where(h_mask, mm, 0.0)
            acc_position += tl.where(h_mask, grad, 0.0)

            token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
            token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_, token_raw)
            token_value = tl.where(token_raw == -1, 0.0, grad)
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
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
        n = tl.arange(0, BLOCK_N)
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


def oracle_embedding_scatter_reduce(
    mm_288: torch.Tensor,
    arg2_1: torch.Tensor,
    arg221_1: torch.Tensor,
    arg836_1: torch.Tensor,
    add_215: torch.Tensor,
    arg220_1: torch.Tensor,
    arg219_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm_288.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    assert mm_288.shape == (ROWS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg221_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg836_1.shape == (BATCH, SEQ, 1)
    assert add_215.shape == (BATCH, SEQ, HIDDEN)
    assert arg220_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg219_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm.shape == (VOCAB, HIDDEN)

    device = mm_288.device
    partial_sum_xhat = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_plain = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out_sum_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_plain = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = mm.clone()
    position_out.zero_()

    _seq_layernorm_scatter_kernel[(SEQ,)](
        mm_288,
        arg2_1,
        arg221_1,
        arg836_1,
        add_215,
        arg220_1,
        arg219_1,
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
        BLOCK_H=triton.next_power_of_2(HIDDEN),
        DROPOUT_SCALE_=DROPOUT_SCALE,
        num_warps=8,
    )
    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, BLOCK_SUM_H),)](
        partial_sum_xhat,
        partial_sum_plain,
        out_sum_xhat,
        out_sum_plain,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_N=triton.next_power_of_2(SEQ),
        BLOCK_H=BLOCK_SUM_H,
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


@oracle_impl(hardware="H100", shapes="(T([2048, 1280], f32), T([1280], f32), T([4, 512, 1280], f32), T([4, 512, 1], f32), T([4, 512, 1280], f32), T([4, 512, 1280], b8), T([1, 512], i64, gen=Index(1024)), T([4, 512], i64, gen=Index(50257)), T([50257, 1280], f32), S([4, 512, 1280]))")
def oracle_forward(inputs):
    return oracle_embedding_scatter_reduce(*inputs)


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
