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

sys.path.insert(0, str(REPO_ROOT))


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


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs(torch.device("cuda"))
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = oracle_embedding_scatter_reduce(*inputs)
        synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok} "
            f"allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(warmup: int, rep: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs(torch.device("cuda"))
    with torch.no_grad():
        oracle_embedding_scatter_reduce(*inputs)
        synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    dense_bytes = VOCAB * HIDDEN * 4
    scatter_bytes = (POSITION_ROWS * HIDDEN + TOKENS * HIDDEN * 2 + HIDDEN * 2) * 4
    print(
        f"oracle_embedding_scatter_reduce: {oracle_us:.3f} us "
        f"shape={SHAPE_LABEL} dense_output={dense_bytes / 1e6:.1f} MB "
        f"scatter_work={scatter_bytes / 1e6:.1f} MB warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare every oracle output against Repro.forward")
    parser.add_argument("--bench", action="store_true", help="time the Triton oracle")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
