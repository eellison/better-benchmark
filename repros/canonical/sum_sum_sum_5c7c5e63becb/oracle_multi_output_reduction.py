"""
Full-scope oracle for sum_sum_sum_5c7c5e63becb (Blenderbot LN/dropout backward).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle consumes the
same inputs as repro.py and returns the same four outputs, including the two
source column reductions, the materialized dropout-scaled layer-norm backward
side output returned as a [2560, 2048] transpose view, and the downstream
column reduction over that side output. It differs from Inductor by doing the
row-local layer-norm reductions, side-output store, and sibling column
accumulators in one row-split Triton producer, then finalizing the column
partials; Inductor cannot do this today because its scheduler represents the
hidden-dimension row reductions, batch/sequence column reductions, dropout
epilogue, view/permute side output, and dependent final reduction as separate
generic nodes rather than one coordinated dependent multi-output reduction.
The fix is COOPERATIVE_SPLIT_K support for dependent multi-output reductions
with materialized side outputs, so row tiles can share row scalar accumulators
and cooperatively reduce all compatible column outputs without rereading the
side-output buffer.
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


REPRO_ID = "sum_sum_sum_5c7c5e63becb"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ = 128
ROWS = BATCH * SEQ
CHANNELS = 2560
DROPOUT_SCALE = 1.1111111111111112
HISTORICAL_BEST_COMPILE_US = 38.72000053524971
ROW_SPLIT = 16
FINALIZE_BLOCK_C = 32


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


@triton.jit
def _row_block_full_scope_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    residual_ptr,
    keep_ptr,
    out_base_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_C], dtype=tl.float32)

    for local_row in tl.range(0, ROW_SPLIT):
        row = row_block * ROW_SPLIT + local_row
        row_mask = row < ROWS_
        mask = row_mask & col_mask
        offsets = row * CHANNELS_ + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        gate = tl.load(gate_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=0)

        out = residual + gate * (weighted * CHANNELS_ - row_sum - rhs * row_dot)
        out = out * keep * DROPOUT_SCALE_

        tl.store(out_base_ptr + offsets, out, mask=mask)
        acc_x_rhs += tl.where(mask, x * rhs, 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_out += tl.where(mask, out, 0.0)

    partial_offsets = row_block * CHANNELS_ + cols
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)
    tl.store(partial_out_ptr + partial_offsets, acc_out, mask=col_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_sum_out_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    col_mask = cols < CHANNELS_
    block_mask = blocks < NUM_ROW_BLOCKS
    mask = block_mask[:, None] & col_mask[None, :]
    offsets = blocks[:, None] * CHANNELS_ + cols[None, :]

    x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_sum_out_ptr + cols, tl.sum(out, axis=0), mask=col_mask)


def oracle_fused(
    mm_2: torch.Tensor,
    arg11_1: torch.Tensor,
    arg34_1: torch.Tensor,
    arg39_1: torch.Tensor,
    arg43_1: torch.Tensor,
    arg33_1: torch.Tensor,
    *_shape_params: object,
    row_split: int = ROW_SPLIT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_2.shape == (ROWS, CHANNELS)
    assert arg11_1.shape == (CHANNELS,)
    assert arg34_1.shape == (BATCH, SEQ, CHANNELS)
    assert arg39_1.shape == (BATCH, SEQ, 1)
    assert arg43_1.shape == (BATCH, SEQ, CHANNELS)
    assert arg33_1.shape == (BATCH, SEQ, CHANNELS)

    rhs = arg34_1.reshape(ROWS, CHANNELS)
    gate = arg39_1.reshape(ROWS)
    residual = arg43_1.reshape(ROWS, CHANNELS)
    keep = arg33_1.reshape(ROWS, CHANNELS)
    device = mm_2.device

    num_row_blocks = triton.cdiv(ROWS, row_split)
    partial_x_rhs = torch.empty((num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    partial_out = torch.empty((num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    out_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    _row_block_full_scope_kernel[(num_row_blocks,)](
        mm_2,
        arg11_1,
        rhs,
        gate,
        residual,
        keep,
        out_base,
        partial_x_rhs,
        partial_x,
        partial_out,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        ROW_SPLIT=row_split,
        BLOCK_C=4096,
        num_warps=16,
    )

    out_x_rhs = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(CHANNELS, FINALIZE_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_out,
        out_x_rhs,
        out_x,
        out_sum_out,
        NUM_ROW_BLOCKS=num_row_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=FINALIZE_BLOCK_C,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_base.permute(1, 0), out_sum_out


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    """Same inputs and outputs as Repro.forward; no subset timing."""
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = len(actual) == len(ref)
    if not ok:
        print(f"SCOPE_MISMATCH: oracle outputs={len(actual)} eager outputs={len(ref)}")

    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        shape_ok = got.shape == expected.shape
        dtype_ok = got.dtype == expected.dtype
        stride_ok = got.stride() == expected.stride()
        ok = ok and output_ok and shape_ok and dtype_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"dtype={got.dtype} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _bench_compile_config(
    label: str,
    config: dict[str, object],
    inputs: tuple[object, ...],
    warmup: int,
    rep: int,
) -> float:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        compiled = _compile_with_config(model, inputs, config)
        compiled_us = triton.testing.do_bench(
            lambda: compiled(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"torch.compile {label}: {compiled_us:.3f} us")
    return compiled_us


def run_bench(rep: int, warmup: int, no_compile: bool) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_forward(inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_fused full-scope LN/dropout multi-output reduction: {oracle_us:.3f} us")

    compile_results: dict[str, float] = {}
    if not no_compile:
        compile_configs = [
            ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
            (
                "combo_looped_cd",
                {
                    "combo_kernels": True,
                    "combo_kernel_per_subkernel_blocks": True,
                    "coordinate_descent_tuning": True,
                    "benchmark_combo_kernel": True,
                    "triton.multi_kernel": 3,
                },
            ),
        ]
        for label, config in compile_configs:
            compile_results[label] = _bench_compile_config(label, config, inputs, warmup, rep)

    best_local_compile_us = min(compile_results.values()) if compile_results else None
    best_required_us = min(
        [x for x in [best_local_compile_us, HISTORICAL_BEST_COMPILE_US] if x is not None]
    )
    true_floor = oracle_us <= best_required_us
    status = "GOOD" if true_floor else "DIAGNOSIS_ONLY"
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": oracle_us,
        "compile_us": best_local_compile_us,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "ratio": (best_local_compile_us / oracle_us) if best_local_compile_us else None,
        "classification": "COOPERATIVE_SPLIT_K",
        "true_floor": true_floor,
        "status": status,
        "compile_configs": compile_results,
    }
    print(json.dumps(result, sort_keys=True))
    if not true_floor:
        print("note: oracle is diagnosis-only because it does not beat all required compile baselines")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
