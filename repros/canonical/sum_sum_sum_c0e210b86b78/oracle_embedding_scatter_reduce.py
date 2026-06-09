"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete NanoGPT layernorm-backward and embedding-gradient return tuple by fusing each token row's hidden reduction with both duplicate-index scatter-add destinations while a tiled Triton outer-product kernel writes the dense vocabulary-gradient side output in its required contiguous layout, whereas Inductor currently lowers the row reductions, pointwise layernorm-backward producer, two `index_put(accumulate=True)` scatter consumers, and dense outer-product add as generic scheduled kernels around materialized intermediates; Inductor cannot do this today because scheduler/codegen does not model a shared rowwise reduction producer feeding multiple structured row scatter-reduces plus a dense add epilogue; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that computes each row producer once, accumulates duplicate row indices directly into every destination, and fuses compatible sibling column reductions and dense side-output initialization."""
from __future__ import annotations

import argparse
import importlib.util
import math
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

REPRO_ID = "sum_sum_sum_c0e210b86b78"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_nanogpt_train_001_4e28ca44"

TOKENS = 64
HIDDEN = 768
VOCAB = 50304
POSITION_ROWS = 1024
BLOCK_HIDDEN = 1024
OUTER_BLOCK_V = 16
OUTER_BLOCK_H = 64
SUM_BLOCK_H = 16
ZERO_BLOCK = 1024



if triton is not None:

    @triton.jit
    def _outer_product_kernel(
        view_ptr,
        arg234_ptr,
        out_ptr,
        view_stride_v: tl.constexpr,
        arg234_stride_h: tl.constexpr,
        VOCAB_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_h = tl.program_id(1)
        v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (v[:, None] < VOCAB_) & (h[None, :] < HIDDEN_)

        lhs = tl.load(view_ptr + v * view_stride_v, mask=v < VOCAB_, other=0.0).to(tl.float32)
        rhs = tl.load(arg234_ptr + h * arg234_stride_h, mask=h < HIDDEN_, other=0.0).to(tl.float32)
        values = lhs[:, None] * rhs[None, :]
        tl.store(out_ptr + v[:, None] * HIDDEN_ + h[None, :], values, mask=mask)

    @triton.jit
    def _zero_position_kernel(
        out2_ptr,
        TOTAL_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL_
        tl.store(out2_ptr + offsets, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)

    @triton.jit
    def _hidden_sums_kernel(
        mm_ptr,
        arg76_ptr,
        out0_ptr,
        out1_ptr,
        HIDDEN_: tl.constexpr,
        TOKENS_: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        t = tl.arange(0, BLOCK_T)
        h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (t[:, None] < TOKENS_) & (h[None, :] < HIDDEN_)
        offsets = t[:, None] * HIDDEN_ + h[None, :]

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        saved = tl.load(arg76_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_weighted = tl.sum(mm * saved, axis=0)
        sum_plain = tl.sum(mm, axis=0)
        tl.store(out0_ptr + h, sum_weighted, mask=h < HIDDEN_)
        tl.store(out1_ptr + h, sum_plain, mask=h < HIDDEN_)

    @triton.jit
    def _row_layernorm_scatter_kernel(
        mm_ptr,
        arg2_ptr,
        arg76_ptr,
        arg259_ptr,
        add70_ptr,
        arg75_ptr,
        arg0_ptr,
        out2_ptr,
        out3_ptr,
        HIDDEN_: tl.constexpr,
        VOCAB_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)
        mask = h < HIDDEN_
        offsets = row * HIDDEN_ + h

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(arg2_ptr + h, mask=mask, other=0.0).to(tl.float32)
        saved = tl.load(arg76_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add70 = tl.load(add70_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(arg259_ptr + row).to(tl.float32)

        weighted = mm * gamma
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, weighted * saved, 0.0), axis=0)
        grad = add70 + scale * (weighted * HIDDEN_ - row_sum - saved * row_dot)

        pos_raw = tl.load(arg75_ptr + row).to(tl.int64)
        pos_valid = (pos_raw >= 0) & (pos_raw < POSITION_ROWS_)
        tl.atomic_add(
            out2_ptr + pos_raw * HIDDEN_ + h,
            grad,
            sem="relaxed",
            mask=mask & pos_valid,
        )

        token_raw = tl.load(arg0_ptr + row).to(tl.int64)
        token_valid = (token_raw >= 0) & (token_raw < VOCAB_)
        tl.atomic_add(
            out3_ptr + token_raw * HIDDEN_ + h,
            grad,
            sem="relaxed",
            mask=mask & token_valid,
        )


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
    view: torch.Tensor,
    arg234_1: torch.Tensor,
    mm_95: torch.Tensor,
    arg2_1: torch.Tensor,
    arg76_1: torch.Tensor,
    arg259_1: torch.Tensor,
    add_70: torch.Tensor,
    arg75_1: torch.Tensor,
    arg0_1: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if view.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    assert view.shape == (1, VOCAB)
    assert arg234_1.shape == (1, HIDDEN)
    assert mm_95.shape == (TOKENS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg76_1.shape == (1, TOKENS, HIDDEN)
    assert arg259_1.shape == (1, TOKENS, 1)
    assert add_70.shape == (1, TOKENS, HIDDEN)
    assert arg75_1.shape == (1, TOKENS)
    assert arg0_1.shape == (1, TOKENS)

    out0 = torch.empty((HIDDEN,), device=view.device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=view.device, dtype=torch.float32)
    out2 = torch.empty((POSITION_ROWS, HIDDEN), device=view.device, dtype=torch.float32)
    out3 = torch.empty((VOCAB, HIDDEN), device=view.device, dtype=torch.float32)

    _outer_product_kernel[(triton.cdiv(VOCAB, OUTER_BLOCK_V), triton.cdiv(HIDDEN, OUTER_BLOCK_H))](
        view,
        arg234_1,
        out3,
        view_stride_v=view.stride(1),
        arg234_stride_h=arg234_1.stride(1),
        VOCAB_=VOCAB,
        HIDDEN_=HIDDEN,
        BLOCK_V=OUTER_BLOCK_V,
        BLOCK_H=OUTER_BLOCK_H,
        num_warps=4,
    )
    _zero_position_kernel[(triton.cdiv(POSITION_ROWS * HIDDEN, ZERO_BLOCK),)](
        out2,
        TOTAL_=POSITION_ROWS * HIDDEN,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _hidden_sums_kernel[(triton.cdiv(HIDDEN, SUM_BLOCK_H),)](
        mm_95,
        arg76_1,
        out0,
        out1,
        HIDDEN_=HIDDEN,
        TOKENS_=TOKENS,
        BLOCK_T=TOKENS,
        BLOCK_H=SUM_BLOCK_H,
        num_warps=4,
    )
    _row_layernorm_scatter_kernel[(TOKENS,)](
        mm_95,
        arg2_1,
        arg76_1,
        arg259_1,
        add_70,
        arg75_1,
        arg0_1,
        out2,
        out3,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        POSITION_ROWS_=POSITION_ROWS,
        BLOCK_H=BLOCK_HIDDEN,
        num_warps=8,
    )
    return out0, out1, out2, out3


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def synchronize() -> None:
    torch.cuda.synchronize()


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


@oracle_impl(hardware="H100", shapes="(T([1, 50304], f32), T([1, 768], f32), T([64, 768], f32), T([768], f32), T([1, 64, 768], f32), T([1, 64, 1], f32), T([1, 64, 768], f32), T([1, 64], i64, gen=Index(1024)), T([1, 64], i64, gen=Index(50304)), S([1, 64, 768]))")
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
