"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete BERT layernorm-backward/dropout return tuple by sharing one rowwise Triton producer for the two returned hidden reductions and the two duplicate-index `index_put(accumulate=True)` embedding-gradient outputs, whereas Inductor currently materializes the `[128,128,768]` gradient producer and lowers the sibling reductions, dropout epilogue, special index-zero guards, and indexed accumulators as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that keeps row-local reduction scalars live while feeding multiple indexed accumulator destinations with per-index fallback values; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that computes each token-row producer once, emits compatible hidden-column reductions, and atomically accumulates all indexed embedding-gradient outputs directly."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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

REPRO_ID = "sum_sum_sum_b3091402bb28"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_bert_pytorch_train_001_b6788913"

BATCH = 128
SEQ = 128
HIDDEN = 768
ROWS = BATCH * SEQ
SEGMENT_ROWS = 3
VOCAB_ROWS = 20005
EPS = 1.0e-6
EXTRA_SCALE = 0.002607561929595828
DROPOUT_SCALE = 1.1111111111111112
INIT_BLOCK = 1024
HIDDEN_BLOCK = 1024



if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        out_sum_x_ptr,
        out_sum_scaled_xhat_ptr,
        out_segment_ptr,
        out_vocab_ptr,
        TOTAL_SEGMENT_: tl.constexpr,
        TOTAL_VOCAB_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        zeros = tl.zeros([BLOCK_], dtype=tl.float32)

        hidden_mask = offsets < HIDDEN_
        segment_mask = offsets < TOTAL_SEGMENT_
        vocab_mask = offsets < TOTAL_VOCAB_
        tl.store(out_sum_x_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(out_sum_scaled_xhat_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(out_segment_ptr + offsets, zeros, mask=segment_mask)
        tl.store(out_vocab_ptr + offsets, zeros, mask=vocab_mask)


    @triton.jit
    def _row_scatter_reduce_kernel(
        mm142_ptr,
        mm144_ptr,
        mm146_ptr,
        gamma_ptr,
        xhat_ptr,
        denom_base_ptr,
        residual_ptr,
        full_ptr,
        dropout_mask_ptr,
        segment_index_ptr,
        token_index_ptr,
        out_sum_x_ptr,
        out_sum_scaled_xhat_ptr,
        out_segment_ptr,
        out_vocab_ptr,
        HIDDEN_: tl.constexpr,
        SEGMENT_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        EPS_: tl.constexpr,
        EXTRA_SCALE_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        mask = h < HIDDEN_
        offsets = row * HIDDEN_ + h

        x = (
            tl.load(mm142_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm144_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        x += tl.load(mm146_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + h, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        denom_raw = tl.load(denom_base_ptr + row).to(tl.float32)
        denom = denom_raw + EPS_

        scaled = x / denom
        gamma_scaled = scaled * gamma
        sum_neg_gamma_scaled = tl.sum(tl.where(mask, -gamma_scaled, 0.0), axis=0)

        gamma_xhat_over_denom2 = gamma * xhat / (denom * denom)
        sum_curvature = tl.sum(
            tl.where(mask, -x * gamma_xhat_over_denom2, 0.0),
            axis=0,
        )

        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        correction = tl.where(
            denom_raw == 0.0,
            tl.load(full_ptr).to(tl.float32),
            sum_curvature / (denom_raw * 2.0),
        )
        grad = residual + gamma_scaled
        grad += correction * EXTRA_SCALE_ * xhat
        grad += sum_neg_gamma_scaled / HIDDEN_

        keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        dropped = grad * keep * DROPOUT_SCALE_

        tl.atomic_add(out_sum_x_ptr + h, x, sem="relaxed", mask=mask)
        tl.atomic_add(
            out_sum_scaled_xhat_ptr + h,
            scaled * xhat,
            sem="relaxed",
            mask=mask,
        )

        full_value = tl.load(full_ptr).to(tl.float32)

        segment_raw = tl.load(segment_index_ptr + row).to(tl.int64)
        segment_wrapped = tl.where(
            segment_raw < 0,
            segment_raw + SEGMENT_ROWS_,
            segment_raw,
        )
        segment_value = tl.where(segment_raw == 0, full_value, dropped)
        tl.atomic_add(
            out_segment_ptr + segment_wrapped * HIDDEN_ + h,
            segment_value,
            sem="relaxed",
            mask=mask & (segment_wrapped >= 0) & (segment_wrapped < SEGMENT_ROWS_),
        )

        token_raw = tl.load(token_index_ptr + row).to(tl.int64)
        token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
        token_value = tl.where(token_raw == 0, full_value, dropped)
        tl.atomic_add(
            out_vocab_ptr + token_wrapped * HIDDEN_ + h,
            token_value,
            sem="relaxed",
            mask=mask & (token_wrapped >= 0) & (token_wrapped < VOCAB_ROWS_),
        )


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    inputs = module.make_inputs()
    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def oracle_bert_embedding_scatter_reduce(
    mm_142: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    arg2_1: torch.Tensor,
    arg103_1: torch.Tensor,
    arg102_1: torch.Tensor,
    add_138: torch.Tensor,
    full_1: torch.Tensor,
    arg101_1: torch.Tensor,
    arg1_1: torch.Tensor,
    arg0_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm_142.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 0: {_shape_param_0}")
    if list(_shape_param_1) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 1: {_shape_param_1}")
    if list(_shape_param_2) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 2: {_shape_param_2}")
    if list(_shape_param_3) != [HIDDEN]:
        raise ValueError(f"unexpected shape parameter 3: {_shape_param_3}")
    if list(_shape_param_4) != [HIDDEN]:
        raise ValueError(f"unexpected shape parameter 4: {_shape_param_4}")
    if list(_shape_param_5) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 5: {_shape_param_5}")

    mm_142 = mm_142.contiguous()
    mm_144 = mm_144.contiguous()
    mm_146 = mm_146.contiguous()
    arg2_1 = arg2_1.contiguous()
    arg103_1 = arg103_1.contiguous()
    arg102_1 = arg102_1.contiguous()
    add_138 = add_138.contiguous()
    full_1 = full_1.contiguous()
    arg101_1 = arg101_1.contiguous()
    arg1_1 = arg1_1.contiguous()
    arg0_1 = arg0_1.contiguous()

    assert mm_142.shape == (ROWS, HIDDEN)
    assert mm_144.shape == (ROWS, HIDDEN)
    assert mm_146.shape == (ROWS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg103_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg102_1.shape == (BATCH, SEQ, 1)
    assert add_138.shape == (BATCH, SEQ, HIDDEN)
    assert full_1.shape == ()
    assert arg101_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg1_1.shape == (BATCH, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)

    device = mm_142.device
    out_sum_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_scaled_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_segment = torch.empty((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    total_segment = SEGMENT_ROWS * HIDDEN
    total_vocab = VOCAB_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(total_vocab, INIT_BLOCK),)](
        out_sum_x,
        out_sum_scaled_xhat,
        out_segment,
        out_vocab,
        TOTAL_SEGMENT_=total_segment,
        TOTAL_VOCAB_=total_vocab,
        HIDDEN_=HIDDEN,
        BLOCK_=INIT_BLOCK,
        num_warps=8,
    )

    _row_scatter_reduce_kernel[(ROWS,)](
        mm_142,
        mm_144,
        mm_146,
        arg2_1,
        arg103_1,
        arg102_1,
        add_138,
        full_1,
        arg101_1,
        arg1_1,
        arg0_1,
        out_sum_x,
        out_sum_scaled_xhat,
        out_segment,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        EPS_=EPS,
        EXTRA_SCALE_=EXTRA_SCALE,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )
    return out_sum_x, out_sum_scaled_xhat, out_segment, out_vocab


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([768], f32), T([128, 128, 768], f32), T([128, 128, 1], f32), T([128, 128, 768], f32), T([], f32), T([128, 128, 768], b8), T([128, 128], i64, gen=Index(3)), T([128, 128], i64, gen=Index(20005)), S([128, 128, 768]), S([128, 128, 768]), S([128, 128, 768]), S([768]), S([768]), S([128, 128, 768]))")
def oracle_forward(inputs):
    return oracle_bert_embedding_scatter_reduce(*inputs)


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
