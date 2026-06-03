"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete T5 dual RMSNorm/dropout backward return tuple, including both hidden-weight reductions and the shared vocabulary-gradient index_put(accumulate=True) side output from rowwise Triton scatter-reduce kernels, whereas Inductor currently materializes each branch's dropout/RMSNorm intermediate, runs the two hidden reductions, and lowers both embedding index_put accumulations as separate generic kernels around the dense vocabulary add; Inductor cannot do this today because scheduler/codegen does not model one-dimensional token-index embedding scatters with duplicate rows as structured scatter-reduce producers that share rowwise RMSNorm reductions and atomically update a common side output; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that computes the row RMSNorm/dropout producer once per token, emits sibling hidden reductions, and accumulates duplicate token rows directly into the initialized vocabulary-gradient output."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_5a4f87bbd879"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_hf_t5_train_001_45c03c17"

BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 512
VOCAB = 32128
DROPOUT_SCALE = 1.1111111111111112
INIT_BLOCK = 1024
HIDDEN_BLOCK = 512

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
    def _branch_rmsnorm_scatter_reduce_kernel(
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
        module = _load_repro_module()
        inputs = module.make_inputs()

    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def oracle_t5_embedding_scatter_reduce(
    mm_117: torch.Tensor,
    mm_119: torch.Tensor,
    mm_121: torch.Tensor,
    arg52_1: torch.Tensor,
    arg221_1: torch.Tensor,
    arg220_1: torch.Tensor,
    arg222_1: torch.Tensor,
    add_61: torch.Tensor,
    arg219_1: torch.Tensor,
    full_1: torch.Tensor,
    mm: torch.Tensor,
    mm_189: torch.Tensor,
    mm_191: torch.Tensor,
    mm_193: torch.Tensor,
    arg2_1: torch.Tensor,
    arg133_1: torch.Tensor,
    arg131_1: torch.Tensor,
    arg134_1: torch.Tensor,
    add_104: torch.Tensor,
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

    assert mm_117.shape == (ROWS, HIDDEN)
    assert mm_119.shape == (ROWS, HIDDEN)
    assert mm_121.shape == (ROWS, HIDDEN)
    assert arg52_1.shape == (HIDDEN,)
    assert arg221_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg220_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg222_1.shape == (BATCH, SEQ, 1)
    assert add_61.shape == (BATCH, SEQ, HIDDEN)
    assert arg219_1.shape == (BATCH, SEQ)
    assert mm.shape == (VOCAB, HIDDEN)
    assert mm_189.shape == (ROWS, HIDDEN)
    assert mm_191.shape == (ROWS, HIDDEN)
    assert mm_193.shape == (ROWS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg133_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg131_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg134_1.shape == (BATCH, SEQ, 1)
    assert add_104.shape == (BATCH, SEQ, HIDDEN)
    assert arg0_1.shape == (BATCH, SEQ)

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

    _branch_rmsnorm_scatter_reduce_kernel[(ROWS,)](
        mm_117,
        mm_119,
        mm_121,
        arg52_1,
        arg221_1,
        arg220_1,
        arg222_1,
        add_61,
        arg219_1,
        full_1,
        out0,
        out_vocab,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )

    _branch_rmsnorm_scatter_reduce_kernel[(ROWS,)](
        mm_189,
        mm_191,
        mm_193,
        arg2_1,
        arg133_1,
        arg131_1,
        arg134_1,
        add_104,
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


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device=device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_t5_embedding_scatter_reduce(*inputs)
        synchronize(device)

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


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def run_bench(device: torch.device, warmup: int, rep: int, compare_eager: bool) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    logical_vocab_bytes = VOCAB * HIDDEN * 4
    logical_branch_bytes = ROWS * HIDDEN * 4

    with torch.no_grad():
        oracle_t5_embedding_scatter_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_t5_embedding_scatter_reduce(*inputs),
            device,
            warmup,
            rep,
        )

    print(
        f"oracle_t5_embedding_scatter_reduce: {oracle_us:.3f} us "
        f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep} "
        f"dense_output={logical_vocab_bytes / 1e6:.1f} MB "
        f"branch_tensor={logical_branch_bytes / 1e6:.1f} MB"
    )

    if not compare_eager:
        return

    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device=device)
    with torch.no_grad():
        model(*inputs)
        synchronize(device)
        eager_us = benchmark(lambda: model(*inputs), device, warmup, rep)
    print(
        f"repro_eager: {eager_us:.3f} us "
        f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs against repro.py")
    parser.add_argument("--bench", action="store_true", help="time the full-scope Triton oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--compare-eager",
        action="store_true",
        help="also benchmark the captured eager repro",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(
            device=device,
            warmup=args.warmup,
            rep=args.rep,
            compare_eager=args.compare_eager,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
