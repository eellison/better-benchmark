"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete MegatronBERT embedding/layernorm-backward return tuple, including the two pre-dropout hidden reductions plus position, segment, and vocabulary index_put(accumulate=True) outputs from one rowwise Triton scatter-reduce producer, whereas Inductor currently materializes the full `[16,512,1024]` row gradient and lowers the sibling reductions and three indexed accumulator destinations as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that shares a rowwise layernorm-backward producer across multiple hidden reductions and indexed outputs with different accumulator sizes; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that keeps the row reductions in registers, emits the hidden reductions, and atomically accumulates the position, segment, and vocabulary gradients directly."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps non-runtime tooling usable.
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

REPRO_ID = "sum_sum_sum_18cf1bb89c04"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ = 512
HIDDEN = 1024
ROWS = BATCH * SEQ
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 29056
DROPOUT_SCALE = 1.1111111111111112
INIT_BLOCK = 1024
HIDDEN_BLOCK = 1024



if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        mm1_ptr,
        out_sum_xhat_ptr,
        out_sum_x_ptr,
        out_position_ptr,
        out_segment_ptr,
        out_vocab_ptr,
        TOTAL_VOCAB_: tl.constexpr,
        TOTAL_POSITION_: tl.constexpr,
        TOTAL_SEGMENT_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)

        vocab_mask = offsets < TOTAL_VOCAB_
        vocab_values = tl.load(mm1_ptr + offsets, mask=vocab_mask, other=0.0).to(tl.float32)
        tl.store(out_vocab_ptr + offsets, vocab_values, mask=vocab_mask)

        position_mask = offsets < TOTAL_POSITION_
        segment_mask = offsets < TOTAL_SEGMENT_
        hidden_mask = offsets < HIDDEN_
        zeros = tl.zeros([BLOCK_], dtype=tl.float32)
        tl.store(out_position_ptr + offsets, zeros, mask=position_mask)
        tl.store(out_segment_ptr + offsets, zeros, mask=segment_mask)
        tl.store(out_sum_xhat_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(out_sum_x_ptr + offsets, zeros, mask=hidden_mask)

    @triton.jit
    def _row_scatter_reduce_kernel(
        mm286_ptr,
        mm288_ptr,
        mm290_ptr,
        weight_ptr,
        saved_x_ptr,
        row_scale_ptr,
        residual_grad_ptr,
        dropout_mask_ptr,
        position_index_ptr,
        full_ptr,
        segment_index_ptr,
        token_index_ptr,
        out_sum_xhat_ptr,
        out_sum_x_ptr,
        out_position_ptr,
        out_segment_ptr,
        out_vocab_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        BATCH_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        SEGMENT_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        hidden_mask = h < HIDDEN_
        base = row * HIDDEN_ + h
        seq = row - (row // SEQ_) * SEQ_

        added = (
            tl.load(mm286_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
            + tl.load(mm288_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        )
        added = added + tl.load(mm290_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        saved_x = tl.load(saved_x_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)

        tl.atomic_add(out_sum_xhat_ptr + h, added * saved_x, sem="relaxed", mask=hidden_mask)
        tl.atomic_add(out_sum_x_ptr + h, added, sem="relaxed", mask=hidden_mask)

        weight = tl.load(weight_ptr + h, mask=hidden_mask, other=0.0).to(tl.float32)
        weighted = added * weight
        row_sum = tl.sum(tl.where(hidden_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(hidden_mask, weighted * saved_x, 0.0), axis=0)
        scale = tl.load(row_scale_ptr + row).to(tl.float32)
        ln_grad = scale * (weighted * HIDDEN_ - row_sum - saved_x * row_dot)

        residual = tl.load(residual_grad_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        keep = tl.load(dropout_mask_ptr + base, mask=hidden_mask, other=0).to(tl.float32)
        grad = (residual + ln_grad) * keep * DROPOUT_SCALE_
        full_value = tl.load(full_ptr).to(tl.float32)

        position_raw = tl.load(position_index_ptr + seq).to(tl.int64)
        position_wrapped = tl.where(position_raw < 0, position_raw + POSITION_ROWS_, position_raw)
        position_value = tl.where(position_raw == -1, full_value / BATCH_, grad)
        tl.atomic_add(
            out_position_ptr + position_wrapped * HIDDEN_ + h,
            position_value,
            sem="relaxed",
            mask=hidden_mask & (position_wrapped >= 0) & (position_wrapped < POSITION_ROWS_),
        )

        segment_raw = tl.load(segment_index_ptr + row).to(tl.int64)
        segment_wrapped = tl.where(segment_raw < 0, segment_raw + SEGMENT_ROWS_, segment_raw)
        tl.atomic_add(
            out_segment_ptr + segment_wrapped * HIDDEN_ + h,
            grad,
            sem="relaxed",
            mask=hidden_mask & (segment_wrapped >= 0) & (segment_wrapped < SEGMENT_ROWS_),
        )

        token_raw = tl.load(token_index_ptr + row).to(tl.int64)
        token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
        token_value = tl.where(token_raw == 0, full_value, grad)
        tl.atomic_add(
            out_vocab_ptr + token_wrapped * HIDDEN_ + h,
            token_value,
            sem="relaxed",
            mask=hidden_mask & (token_wrapped >= 0) & (token_wrapped < VOCAB_ROWS_),
        )


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        module = _load_repro_module()
        inputs = module.make_inputs()

    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def oracle_megatronbert_embedding_scatter_reduce(
    mm_286: torch.Tensor,
    mm_288: torch.Tensor,
    mm_290: torch.Tensor,
    arg3_1: torch.Tensor,
    arg200_1: torch.Tensor,
    arg643_1: torch.Tensor,
    add_143: torch.Tensor,
    arg199_1: torch.Tensor,
    arg2_1: torch.Tensor,
    full_1: torch.Tensor,
    arg198_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm_286.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 0: {_shape_param_0}")
    if list(_shape_param_1) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 1: {_shape_param_1}")
    if list(_shape_param_2) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 2: {_shape_param_2}")

    mm_286 = mm_286.contiguous()
    mm_288 = mm_288.contiguous()
    mm_290 = mm_290.contiguous()
    arg3_1 = arg3_1.contiguous()
    arg200_1 = arg200_1.contiguous()
    arg643_1 = arg643_1.contiguous()
    add_143 = add_143.contiguous()
    arg199_1 = arg199_1.contiguous()
    arg2_1 = arg2_1.contiguous()
    full_1 = full_1.contiguous()
    arg198_1 = arg198_1.contiguous()
    arg0_1 = arg0_1.contiguous()
    mm_1 = mm_1.contiguous()

    assert mm_286.shape == (ROWS, HIDDEN)
    assert mm_288.shape == (ROWS, HIDDEN)
    assert mm_290.shape == (ROWS, HIDDEN)
    assert arg3_1.shape == (HIDDEN,)
    assert arg200_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg643_1.shape == (BATCH, SEQ, 1)
    assert add_143.shape == (BATCH, SEQ, HIDDEN)
    assert arg199_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg2_1.shape == (1, SEQ)
    assert full_1.shape == ()
    assert arg198_1.shape == (BATCH, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm_1.shape == (VOCAB_ROWS, HIDDEN)

    device = mm_286.device
    out_sum_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_segment = torch.empty((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    total_vocab = VOCAB_ROWS * HIDDEN
    total_position = POSITION_ROWS * HIDDEN
    total_segment = SEGMENT_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(total_vocab, INIT_BLOCK),)](
        mm_1,
        out_sum_xhat,
        out_sum_x,
        out_position,
        out_segment,
        out_vocab,
        TOTAL_VOCAB_=total_vocab,
        TOTAL_POSITION_=total_position,
        TOTAL_SEGMENT_=total_segment,
        HIDDEN_=HIDDEN,
        BLOCK_=INIT_BLOCK,
        num_warps=8,
    )
    _row_scatter_reduce_kernel[(ROWS,)](
        mm_286,
        mm_288,
        mm_290,
        arg3_1,
        arg200_1,
        arg643_1,
        add_143,
        arg199_1,
        arg2_1,
        full_1,
        arg198_1,
        arg0_1,
        out_sum_xhat,
        out_sum_x,
        out_position,
        out_segment,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BATCH_=BATCH,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )
    return out_sum_xhat, out_sum_x, out_position, out_segment, out_vocab


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
    return oracle_megatronbert_embedding_scatter_reduce(*inputs)


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
