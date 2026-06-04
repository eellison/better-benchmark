"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete two-branch T5 layernorm/dropout embedding-backward tuple, including both `[768]` hidden reductions and the final `[32128, 768]` duplicate-index vocabulary scatter-add, whereas Inductor currently materializes the `[8, 1024, 768]` rowwise gradient producers and lowers the sibling reductions plus two `index_put(accumulate=True)` scatters as separate generic kernels; Inductor cannot do this today because scheduler/codegen does not model an embedding row scatter-reduce producer that feeds column reductions and multiple accumulated indexed side outputs from the same rowwise layernorm-backward math; the fix is SCATTER_REDUCE: add a structured embedding-backward scatter-reduce lowering that keeps each token row producer live for hidden reductions and direct vocabulary accumulation across both branches."""
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


REPRO_ID = "sum_sum_sum_3e8dba104ec0"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_hf_t5_base_train_001_199c262d"

BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 768
VOCAB = 32128
DROPOUT_SCALE = 1.1111111111111112
INIT_BLOCK = 1024
HIDDEN_BLOCK = 1024

sys.path.insert(0, str(REPO_ROOT))


if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        mm_ptr,
        out0_ptr,
        out1_ptr,
        out_vocab_ptr,
        mm_stride_v: tl.constexpr,
        mm_stride_h: tl.constexpr,
        TOTAL_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        vocab_mask = offsets < TOTAL_
        rows = offsets // HIDDEN_
        hidden = offsets - rows * HIDDEN_
        values = tl.load(
            mm_ptr + rows * mm_stride_v + hidden * mm_stride_h,
            mask=vocab_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(out_vocab_ptr + offsets, values, mask=vocab_mask)

        hidden_mask = offsets < HIDDEN_
        zeros = tl.zeros([BLOCK_], dtype=tl.float32)
        tl.store(out0_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(out1_ptr + offsets, zeros, mask=hidden_mask)

    @triton.jit
    def _branch_layernorm_scatter_reduce_kernel(
        mm0_ptr,
        mm1_ptr,
        mm2_ptr,
        weight_ptr,
        mask_ptr,
        saved_ptr,
        rstd_ptr,
        add_ptr,
        index_ptr,
        full_ptr,
        out_reduce_ptr,
        out_vocab_ptr,
        HIDDEN_: tl.constexpr,
        VOCAB_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        hidden_mask = h < HIDDEN_
        base = row * HIDDEN_ + h

        added = (
            tl.load(mm0_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
            + tl.load(mm1_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
            + tl.load(mm2_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        )
        keep = tl.load(mask_ptr + base, mask=hidden_mask, other=0).to(tl.float32)
        saved = tl.load(saved_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)
        weight = tl.load(weight_ptr + h, mask=hidden_mask, other=0.0).to(tl.float32)
        prior_grad = tl.load(add_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)

        dropped_saved = keep * saved * DROPOUT_SCALE_
        weighted = added * weight
        row_dot = tl.sum(tl.where(hidden_mask, weighted * dropped_saved, 0.0), axis=0)

        reduce_contrib = added * dropped_saved * rstd
        tl.atomic_add(out_reduce_ptr + h, reduce_contrib, sem="relaxed", mask=hidden_mask)

        correction = row_dot * rstd * rstd * rstd * dropped_saved * (-1.0 / HIDDEN_)
        grad = (prior_grad + weighted * rstd + correction) * keep * DROPOUT_SCALE_

        raw_index = tl.load(index_ptr + row).to(tl.int64)
        wrapped_index = tl.where(raw_index < 0, raw_index + VOCAB_, raw_index)
        full_value = tl.load(full_ptr).to(tl.float32)
        scatter_value = tl.where(raw_index == -1, full_value, grad)
        valid_index = (wrapped_index >= 0) & (wrapped_index < VOCAB_)
        tl.atomic_add(
            out_vocab_ptr + wrapped_index * HIDDEN_ + h,
            scatter_value,
            sem="relaxed",
            mask=hidden_mask & valid_index,
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
        inputs = _load_repro_module().make_inputs()

    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def _check_common_inputs(
    mm0: torch.Tensor,
    mm1: torch.Tensor,
    mm2: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    saved: torch.Tensor,
    rstd: torch.Tensor,
    add: torch.Tensor,
    index: torch.Tensor,
) -> None:
    assert mm0.shape == (ROWS, HIDDEN) and mm0.is_contiguous()
    assert mm1.shape == (ROWS, HIDDEN) and mm1.is_contiguous()
    assert mm2.shape == (ROWS, HIDDEN) and mm2.is_contiguous()
    assert weight.shape == (HIDDEN,) and weight.is_contiguous()
    assert mask.shape == (BATCH, SEQ, HIDDEN) and mask.is_contiguous()
    assert saved.shape == (BATCH, SEQ, HIDDEN) and saved.is_contiguous()
    assert rstd.shape == (BATCH, SEQ, 1) and rstd.is_contiguous()
    assert add.shape == (BATCH, SEQ, HIDDEN) and add.is_contiguous()
    assert index.shape == (BATCH, SEQ) and index.is_contiguous()
    assert mm0.dtype == torch.float32 and mm1.dtype == torch.float32 and mm2.dtype == torch.float32
    assert weight.dtype == torch.float32 and saved.dtype == torch.float32
    assert rstd.dtype == torch.float32 and add.dtype == torch.float32
    assert mask.dtype == torch.bool and index.dtype == torch.int64


def oracle_t5_embedding_scatter_reduce(
    mm_237: torch.Tensor,
    mm_239: torch.Tensor,
    mm_241: torch.Tensor,
    arg100_1: torch.Tensor,
    arg425_1: torch.Tensor,
    arg424_1: torch.Tensor,
    arg426_1: torch.Tensor,
    add_127: torch.Tensor,
    arg423_1: torch.Tensor,
    full_1: torch.Tensor,
    mm: torch.Tensor,
    mm_381: torch.Tensor,
    mm_383: torch.Tensor,
    mm_385: torch.Tensor,
    arg2_1: torch.Tensor,
    arg259_1: torch.Tensor,
    arg257_1: torch.Tensor,
    arg260_1: torch.Tensor,
    add_212: torch.Tensor,
    arg0_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
    _shape_param_9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
    )
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    _check_common_inputs(
        mm_237,
        mm_239,
        mm_241,
        arg100_1,
        arg425_1,
        arg424_1,
        arg426_1,
        add_127,
        arg423_1,
    )
    _check_common_inputs(
        mm_381,
        mm_383,
        mm_385,
        arg2_1,
        arg259_1,
        arg257_1,
        arg260_1,
        add_212,
        arg0_1,
    )
    assert full_1.shape == () and full_1.dtype == torch.float32
    assert mm.shape == (VOCAB, HIDDEN) and mm.dtype == torch.float32 and mm.is_contiguous()

    out0 = torch.empty((HIDDEN,), device=mm.device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=mm.device, dtype=torch.float32)
    out_vocab = torch.empty_strided(
        tuple(mm.shape),
        tuple(mm.stride()),
        device=mm.device,
        dtype=mm.dtype,
    )

    _init_outputs_kernel[(triton.cdiv(VOCAB * HIDDEN, INIT_BLOCK),)](
        mm,
        out0,
        out1,
        out_vocab,
        mm_stride_v=mm.stride(0),
        mm_stride_h=mm.stride(1),
        TOTAL_=VOCAB * HIDDEN,
        HIDDEN_=HIDDEN,
        BLOCK_=INIT_BLOCK,
        num_warps=4,
    )

    _branch_layernorm_scatter_reduce_kernel[(ROWS,)](
        mm_237,
        mm_239,
        mm_241,
        arg100_1,
        arg425_1,
        arg424_1,
        arg426_1,
        add_127,
        arg423_1,
        full_1,
        out0,
        out_vocab,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )

    _branch_layernorm_scatter_reduce_kernel[(ROWS,)](
        mm_381,
        mm_383,
        mm_385,
        arg2_1,
        arg259_1,
        arg257_1,
        arg260_1,
        add_212,
        arg0_1,
        full_1,
        out1,
        out_vocab,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )
    return out0, out1, out_vocab


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


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
        actual = oracle_t5_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()

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
            f"shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(warmup: int, rep: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs(torch.device("cuda"))
    with torch.no_grad():
        oracle_t5_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_t5_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    dense_mb = VOCAB * HIDDEN * 4 / 1e6
    branch_mb = ROWS * HIDDEN * 4 / 1e6
    print(
        f"oracle_t5_embedding_scatter_reduce: {oracle_us:.3f} us "
        f"impl=triton shape={SHAPE_LABEL} dense_output={dense_mb:.1f} MB "
        f"branch_tensor={branch_mb:.1f} MB warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs against repro.py")
    parser.add_argument("--bench", action="store_true", help="time the full-scope Triton oracle")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("pass --check and/or --bench")

    ok = True
    if args.check:
        ok = run_check(rtol=args.rtol, atol=args.atol)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
