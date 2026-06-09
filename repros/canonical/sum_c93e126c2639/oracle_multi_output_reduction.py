"""
Full-scope Triton oracle for sum_c93e126c2639.

Gap diagnosis (classification: BANDWIDTH_BOUND): This oracle consumes the same
original inputs as repro.py and returns the same single contiguous
`float32[48, 512, 512]` output. It fuses the dropout mask, attention mask,
softmax-gradient producer, row sum, and dependent FMA into one specialized
Triton row kernel over the complete width-512 attention row, avoiding any
subset-only reduction timing. Inductor cannot do materially better today
because the compiled region is already a single fused kernel for this small-row
attention backward tail; remaining time is dominated by reading the logits,
upstream gradient, two masks, and writing the full tensor. The fix class is
BANDWIDTH_BOUND.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

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



REPRO_ID = "sum_c93e126c2639"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
HEADS = 12
ROWS_PER_HEAD = 512
COLS = 512
TOTAL_ROWS = BATCH * HEADS * ROWS_PER_HEAD
DROPOUT_SCALE = 1.1111111111111112
HISTORICAL_BEST_COMPILE_US = 30.14400042593479

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _attention_backward_row_kernel(
    bmm_ptr,
    dropout_mask_ptr,
    attention_mask_ptr,
    logits_ptr,
    row_max_ptr,
    row_denom_ptr,
    zero_row_ptr,
    out_ptr,
    DROPOUT_SCALE_: tl.constexpr,
    COLS_: tl.constexpr,
    HEADS_: tl.constexpr,
    ROWS_PER_HEAD_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    active = cols < COLS_

    b = row // (HEADS_ * ROWS_PER_HEAD_)
    row_in_batch_heads = row - b * HEADS_ * ROWS_PER_HEAD_
    m = row_in_batch_heads % ROWS_PER_HEAD_
    data_offsets = row * COLS_ + cols
    attention_mask_offsets = (b * ROWS_PER_HEAD_ + m) * COLS_ + cols

    keep_attention = tl.load(attention_mask_ptr + attention_mask_offsets, mask=active, other=0)
    zero_row = tl.load(zero_row_ptr + row)

    logits = tl.load(logits_ptr + data_offsets, mask=active, other=-float("inf")).to(tl.float32)
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + row).to(tl.float32)
    masked_logits = tl.where(keep_attention, logits - row_max, -float("inf"))
    prob = tl.exp(masked_logits) / row_denom
    prob = tl.where(zero_row, 0.0, prob)

    dropout_keep = tl.load(dropout_mask_ptr + data_offsets, mask=active, other=0).to(tl.float32)
    upstream = tl.load(bmm_ptr + data_offsets, mask=active, other=0.0).to(tl.float32)
    scaled_upstream = upstream * dropout_keep * DROPOUT_SCALE_
    weighted = scaled_upstream * prob
    row_sum = tl.sum(tl.where(active, weighted, 0.0), axis=0)

    out = weighted - prob * row_sum
    tl.store(out_ptr + data_offsets, out, mask=active)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def _validate_inputs(inputs: tuple[object, ...]) -> None:
    (
        bmm_1,
        arg14_1,
        arg4_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        shape0,
        shape1,
        shape2,
    ) = inputs
    expected_view = (BATCH, HEADS, ROWS_PER_HEAD, COLS)
    if tuple(shape0) != expected_view or tuple(shape1) != expected_view:
        raise ValueError("unexpected 4D shape parameters")
    if tuple(shape2) != (BATCH * HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError("unexpected output shape parameter")
    if bmm_1.shape != (BATCH * HEADS, ROWS_PER_HEAD, COLS):
        raise ValueError(f"unexpected bmm_1 shape: {tuple(bmm_1.shape)}")
    if arg10_1.shape != bmm_1.shape:
        raise ValueError(f"unexpected arg10_1 shape: {tuple(arg10_1.shape)}")
    if arg14_1.shape != expected_view or arg4_1.shape != (BATCH, 1, ROWS_PER_HEAD, COLS):
        raise ValueError("unexpected mask shapes")
    if arg11_1.shape != (BATCH, HEADS, ROWS_PER_HEAD, 1):
        raise ValueError("unexpected arg11_1 shape")
    if arg12_1.shape != arg11_1.shape or arg13_1.shape != arg11_1.shape:
        raise ValueError("unexpected row scalar shapes")
    tensors = (bmm_1, arg14_1, arg4_1, arg10_1, arg11_1, arg12_1, arg13_1)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("oracle expects the captured contiguous input layout")
    if bmm_1.dtype != torch.float32 or arg10_1.dtype != torch.float32:
        raise ValueError("expected float32 bmm/logit tensors")
    if arg11_1.dtype != torch.float32 or arg12_1.dtype != torch.float32:
        raise ValueError("expected float32 row max/denominator tensors")
    if arg14_1.dtype != torch.bool or arg4_1.dtype != torch.bool or arg13_1.dtype != torch.bool:
        raise ValueError("expected bool masks")


def oracle_fused(
    bmm_1: torch.Tensor,
    arg14_1: torch.Tensor,
    arg4_1: torch.Tensor,
    arg10_1: torch.Tensor,
    arg11_1: torch.Tensor,
    arg12_1: torch.Tensor,
    arg13_1: torch.Tensor,
    shape0,
    shape1,
    shape2,
) -> torch.Tensor:
    inputs = (bmm_1, arg14_1, arg4_1, arg10_1, arg11_1, arg12_1, arg13_1, shape0, shape1, shape2)
    _validate_inputs(inputs)

    out = torch.empty_strided(
        (BATCH * HEADS, ROWS_PER_HEAD, COLS),
        (ROWS_PER_HEAD * COLS, COLS, 1),
        device=bmm_1.device,
        dtype=torch.float32,
    )
    _attention_backward_row_kernel[(TOTAL_ROWS,)](
        bmm_1,
        arg14_1,
        arg4_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        out,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        COLS_=COLS,
        HEADS_=HEADS,
        ROWS_PER_HEAD_=ROWS_PER_HEAD,
        BLOCK_N=COLS,
        num_warps=1,
    )
    return out


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


@oracle_impl(hardware="H100", shapes="(T([48, 512, 512], f32), T([4, 12, 512, 512], b8), T([4, 1, 512, 512], b8), T([48, 512, 512], f32), T([4, 12, 512, 1], f32), T([4, 12, 512, 1], f32), T([4, 12, 512, 1], b8), S([4, 12, 512, 512]), S([4, 12, 512, 512]), S([48, 512, 512]))")
def oracle_forward(inputs):
    return oracle_fused(*inputs)


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
