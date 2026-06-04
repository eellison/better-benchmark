"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer layer-norm-backward/dropout return tuple, including both `[768]` hidden reductions and all three duplicate-index `index_put(accumulate=True)` embedding scatter outputs, whereas Inductor currently materializes the rowwise layer-norm backward producer and lowers the sibling sums plus three indexed scatters as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured embedding scatter-reduce lowering that keeps a row producer live while feeding multiple duplicate-index accumulation outputs and hidden-column reductions; the fix is SCATTER_REDUCE: add an embedding scatter-reduce template that computes each token row once, accumulates compatible column reductions, and atomically scatters masked layer-norm gradients directly into each destination table."""
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


REPRO_ID = "sum_sum_sum_72d5ffa26b42"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_hf_longformer_train_006_e8e0f483"

BATCH = 2
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 768
POSITION_ROWS = 1
GLOBAL_ROWS = 4098
TOKEN_ROWS = 50265
DROPOUT_SCALE = 1.1111111111111112
BLOCK_H = 1024

sys.path.insert(0, str(REPO_ROOT))


if triton is not None:

    @triton.jit
    def _layernorm_scatter_reduce_kernel(
        mask_ptr,
        grad_ptr,
        weight_ptr,
        saved_ptr,
        inv_factor_ptr,
        index0_ptr,
        index1_ptr,
        index2_ptr,
        out_sum_saved_ptr,
        out_sum_plain_ptr,
        out_pos_ptr,
        out_global_ptr,
        out_token_ptr,
        HIDDEN_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        GLOBAL_ROWS_: tl.constexpr,
        TOKEN_ROWS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        hidden_mask = h < HIDDEN_
        base = row * HIDDEN_ + h

        keep = tl.load(mask_ptr + base, mask=hidden_mask, other=0).to(tl.float32)
        grad = tl.load(grad_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + h, mask=hidden_mask, other=0.0).to(tl.float32)
        saved = tl.load(saved_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        inv_factor = tl.load(inv_factor_ptr + row).to(tl.float32)

        dropped = grad * keep * DROPOUT_SCALE_
        weighted = dropped * weight
        row_sum = tl.sum(tl.where(hidden_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(hidden_mask, weighted * saved, 0.0), axis=0)
        dx = inv_factor * (weighted * HIDDEN_ - row_sum - saved * row_dot)

        tl.atomic_add(
            out_sum_saved_ptr + h,
            dropped * saved,
            sem="relaxed",
            mask=hidden_mask,
        )
        tl.atomic_add(
            out_sum_plain_ptr + h,
            dropped,
            sem="relaxed",
            mask=hidden_mask,
        )

        idx0_raw = tl.load(index0_ptr + row).to(tl.int64)
        idx0 = tl.where(idx0_raw < 0, idx0_raw + POSITION_ROWS_, idx0_raw)
        idx0_valid = (idx0 >= 0) & (idx0 < POSITION_ROWS_)
        tl.atomic_add(
            out_pos_ptr + idx0 * HIDDEN_ + h,
            dx,
            sem="relaxed",
            mask=hidden_mask & idx0_valid,
        )

        idx1_raw = tl.load(index1_ptr + row).to(tl.int64)
        idx1 = tl.where(idx1_raw < 0, idx1_raw + GLOBAL_ROWS_, idx1_raw)
        idx1_valid = (idx1 >= 0) & (idx1 < GLOBAL_ROWS_) & (idx1_raw != 1)
        tl.atomic_add(
            out_global_ptr + idx1 * HIDDEN_ + h,
            dx,
            sem="relaxed",
            mask=hidden_mask & idx1_valid,
        )

        idx2_raw = tl.load(index2_ptr + row).to(tl.int64)
        idx2 = tl.where(idx2_raw < 0, idx2_raw + TOKEN_ROWS_, idx2_raw)
        idx2_valid = (idx2 >= 0) & (idx2 < TOKEN_ROWS_) & (idx2_raw != 1)
        tl.atomic_add(
            out_token_ptr + idx2 * HIDDEN_ + h,
            dx,
            sem="relaxed",
            mask=hidden_mask & idx2_valid,
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


def _check_inputs(
    arg5_1: torch.Tensor,
    arg7_1: torch.Tensor,
    arg1_1: torch.Tensor,
    arg4_1: torch.Tensor,
    arg6_1: torch.Tensor,
    arg2_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg0_1: torch.Tensor,
) -> None:
    assert arg5_1.shape == (BATCH, SEQ, HIDDEN) and arg5_1.is_contiguous()
    assert arg7_1.shape == (BATCH, SEQ, HIDDEN) and arg7_1.is_contiguous()
    assert arg1_1.shape == (HIDDEN,) and arg1_1.is_contiguous()
    assert arg4_1.shape == (BATCH, SEQ, HIDDEN) and arg4_1.is_contiguous()
    assert arg6_1.shape == (BATCH, SEQ, 1) and arg6_1.is_contiguous()
    assert arg2_1.shape == (BATCH, SEQ) and arg2_1.is_contiguous()
    assert arg3_1.shape == (BATCH, SEQ) and arg3_1.is_contiguous()
    assert arg0_1.shape == (BATCH, SEQ) and arg0_1.is_contiguous()
    assert arg5_1.dtype == torch.bool
    assert arg7_1.dtype == torch.float32
    assert arg1_1.dtype == torch.float32
    assert arg4_1.dtype == torch.float32
    assert arg6_1.dtype == torch.float32
    assert arg2_1.dtype == torch.int64
    assert arg3_1.dtype == torch.int64
    assert arg0_1.dtype == torch.int64


def oracle_longformer_embedding_scatter_reduce(
    arg5_1: torch.Tensor,
    arg7_1: torch.Tensor,
    arg1_1: torch.Tensor,
    arg4_1: torch.Tensor,
    arg6_1: torch.Tensor,
    arg2_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg0_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if arg7_1.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    _check_inputs(arg5_1, arg7_1, arg1_1, arg4_1, arg6_1, arg2_1, arg3_1, arg0_1)

    device = arg7_1.device
    out_sum_saved = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_plain = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    out_pos = torch.zeros((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_global = torch.zeros((GLOBAL_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_token = torch.zeros((TOKEN_ROWS, HIDDEN), device=device, dtype=torch.float32)

    _layernorm_scatter_reduce_kernel[(ROWS,)](
        arg5_1,
        arg7_1,
        arg1_1,
        arg4_1,
        arg6_1,
        arg2_1,
        arg3_1,
        arg0_1,
        out_sum_saved,
        out_sum_plain,
        out_pos,
        out_global,
        out_token,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        GLOBAL_ROWS_=GLOBAL_ROWS,
        TOKEN_ROWS_=TOKEN_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=BLOCK_H,
        num_warps=8,
    )

    return out_sum_saved, out_sum_plain, out_pos, out_global, out_token


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
        actual = oracle_longformer_embedding_scatter_reduce(*inputs)
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
        oracle_longformer_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_longformer_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    token_mb = TOKEN_ROWS * HIDDEN * 4 / 1e6
    global_mb = GLOBAL_ROWS * HIDDEN * 4 / 1e6
    row_mb = ROWS * HIDDEN * 4 / 1e6
    print(
        f"oracle_longformer_embedding_scatter_reduce: {oracle_us:.3f} us "
        f"impl=triton shape={SHAPE_LABEL} token_output={token_mb:.1f} MB "
        f"global_output={global_mb:.1f} MB row_producer={row_mb:.1f} MB "
        f"warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs against repro.py")
    parser.add_argument("--bench", action="store_true", help="time the full-scope Triton oracle")
    parser.add_argument("--rtol", type=float, default=1e-3)
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
