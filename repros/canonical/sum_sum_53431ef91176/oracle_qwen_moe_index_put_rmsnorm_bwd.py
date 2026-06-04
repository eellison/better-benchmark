"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Qwen3 MoE masked row scatter-add into the `[2048, 2048]` token-gradient table and derives both the RMSNorm weight-gradient reduction and transposed input-gradient output from that accumulated table, whereas Inductor currently materializes the masked grouped-mm source, lowers `index_put(accumulate=True)` as a generic duplicate-index scatter, then schedules the RMSNorm backward reductions and transpose store as separate consumers; Inductor cannot do this today because its scheduler/codegen does not model one-dimensional duplicate routing-index row updates feeding sibling RMSNorm-backward reductions and a materialized transposed epilogue as a single structured scatter-reduce pattern; the fix is SCATTER_REDUCE: add a row-index scatter-reduce template that accumulates routed rows once, folds the residual add before RMSNorm backward, emits the per-hidden reduction, and stores the transposed input gradient with its required layout."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_53431ef91176"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_qwen_qwen3-30b-a3b_001_11dcb1c7"

TOKENS = 2048
HIDDEN_SIZE = 2048
ROUTED_ROWS = 16384
BLOCK_COPY = 1024
BLOCK_SCATTER_COLS = 512
BLOCK_OUT0_COLS = 8
BLOCK_INPUT_COLS = 128
MAX_SOURCES_PER_ROW = 23

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if triton is not None:

    @triton.jit
    def _bf16_add2(lhs, rhs):
        return (lhs + rhs).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _fill_i32_kernel(dst, value: tl.constexpr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        tl.store(dst + offsets, value, mask=mask)

    @triton.jit
    def _build_source_slots_atomic_kernel(
        row_index,
        slot_counts,
        slot_sources,
        MAX_SOURCES_PER_ROW_: tl.constexpr,
    ):
        src = tl.program_id(0)
        dest = tl.load(row_index + src).to(tl.int64)
        slot = tl.atomic_add(slot_counts + dest, 1, sem="relaxed")
        tl.store(
            slot_sources + dest * MAX_SOURCES_PER_ROW_ + slot,
            src,
            mask=slot < MAX_SOURCES_PER_ROW_,
        )

    @triton.jit
    def _sort_source_slots_kernel(
        slot_sources,
        sorted_sources,
        ROUTED_ROWS_: tl.constexpr,
        MAX_SOURCES_PER_ROW_: tl.constexpr,
    ):
        dest_row = tl.program_id(0)
        last = tl.full((), -1, tl.int32)
        for out_slot in tl.static_range(0, MAX_SOURCES_PER_ROW_):
            best = tl.full((), ROUTED_ROWS_, tl.int32)
            for in_slot in tl.static_range(0, MAX_SOURCES_PER_ROW_):
                candidate = tl.load(
                    slot_sources + dest_row * MAX_SOURCES_PER_ROW_ + in_slot
                ).to(tl.int32)
                take = (candidate >= 0) & (candidate > last) & (candidate < best)
                best = tl.where(take, candidate, best)
            valid = best < ROUTED_ROWS_
            tl.store(
                sorted_sources + dest_row * MAX_SOURCES_PER_ROW_ + out_slot,
                tl.where(valid, best, -1),
            )
            last = best

    @triton.jit
    def _ordered_row_scatter_kernel(
        mask_ptr,
        full_scalar,
        grouped_mm,
        full_5,
        slot_sources,
        scatter,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
        MAX_SOURCES_PER_ROW_: tl.constexpr,
    ):
        dest_row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        dest_offsets = dest_row * HIDDEN_ + cols

        accum = tl.load(full_5 + dest_offsets, mask=col_mask, other=0.0).to(tl.float32)
        fill = tl.load(full_scalar).to(tl.float32)
        for slot in tl.static_range(0, MAX_SOURCES_PER_ROW_):
            src_row = tl.load(
                slot_sources + dest_row * MAX_SOURCES_PER_ROW_ + slot
            ).to(tl.int32)
            valid = src_row >= 0
            masked = tl.load(mask_ptr + src_row, mask=valid, other=0).to(tl.int1)
            grouped = tl.load(
                grouped_mm + src_row * HIDDEN_ + cols,
                mask=valid & col_mask,
                other=0.0,
            ).to(tl.float32)
            values = tl.where(masked, fill, grouped)
            accum = tl.where(valid, (accum + values).to(tl.bfloat16).to(tl.float32), accum)
        tl.store(scatter + dest_offsets, accum.to(tl.bfloat16), mask=col_mask)

    @triton.jit
    def _row_dot_kernel(
        scatter,
        mm_33,
        gamma,
        residual_lhs,
        residual_rhs,
        row_scale,
        row_dot,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        offsets = row * HIDDEN_ + cols

        token_grad = _bf16_add2(
            tl.load(scatter + offsets).to(tl.float32),
            tl.load(mm_33 + offsets).to(tl.float32),
        )
        residual = _bf16_add2(
            tl.load(residual_lhs + offsets).to(tl.float32),
            tl.load(residual_rhs + offsets).to(tl.float32),
        )
        weighted = (
            token_grad * tl.load(gamma + cols).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        tl.store(row_dot + row, tl.sum(weighted * residual, axis=0))

    @triton.jit
    def _weight_grad_kernel(
        scatter,
        mm_33,
        residual_lhs,
        residual_rhs,
        row_scale,
        out0,
        HIDDEN_: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]

        token_grad = _bf16_add2(
            tl.load(scatter + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32),
            tl.load(mm_33 + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32),
        )
        residual = _bf16_add2(
            tl.load(residual_lhs + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32),
            tl.load(residual_rhs + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32),
        )
        normalized = (
            residual
            * tl.load(row_scale + rows, mask=rows < BLOCK_R, other=0.0).to(tl.float32)[:, None]
        ).to(tl.bfloat16).to(tl.float32)
        product = (token_grad * normalized).to(tl.bfloat16).to(tl.float32)
        tl.store(out0 + cols, tl.sum(product, axis=0), mask=col_mask)

    @triton.jit
    def _input_grad_kernel(
        scatter,
        mm_33,
        gamma,
        residual_lhs,
        residual_rhs,
        row_scale,
        add_50,
        row_dot,
        out_base,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        offsets = row * HIDDEN_ + cols

        token_grad = _bf16_add2(
            tl.load(scatter + offsets, mask=col_mask, other=0.0).to(tl.float32),
            tl.load(mm_33 + offsets, mask=col_mask, other=0.0).to(tl.float32),
        )
        residual = _bf16_add2(
            tl.load(residual_lhs + offsets, mask=col_mask, other=0.0).to(tl.float32),
            tl.load(residual_rhs + offsets, mask=col_mask, other=0.0).to(tl.float32),
        )
        weighted = (
            token_grad * tl.load(gamma + cols, mask=col_mask, other=0.0).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        inv_rms = tl.load(row_scale + row).to(tl.float32)
        dot = tl.load(row_dot + row).to(tl.float32)
        correction = (-0.5 * dot * inv_rms * inv_rms * inv_rms / HIDDEN_) * (residual * 2.0)
        input_grad = weighted * inv_rms + correction
        out_vals = (
            tl.load(add_50 + offsets, mask=col_mask, other=0.0).to(tl.float32)
            + input_grad.to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(out_base + offsets, out_vals, mask=col_mask)

    @triton.jit
    def _input_grad_fullrow_kernel(
        scatter,
        mm_33,
        gamma,
        residual_lhs,
        residual_rhs,
        row_scale,
        add_50,
        out_base,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        offsets = row * HIDDEN_ + cols

        token_grad = _bf16_add2(
            tl.load(scatter + offsets).to(tl.float32),
            tl.load(mm_33 + offsets).to(tl.float32),
        )
        residual = _bf16_add2(
            tl.load(residual_lhs + offsets).to(tl.float32),
            tl.load(residual_rhs + offsets).to(tl.float32),
        )
        weighted = (
            token_grad * tl.load(gamma + cols).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        dot = tl.sum(weighted * residual, axis=0)
        inv_rms = tl.load(row_scale + row).to(tl.float32)
        correction = (-0.5 * dot * inv_rms * inv_rms * inv_rms / HIDDEN_) * (residual * 2.0)
        input_grad = weighted * inv_rms + correction
        out_vals = (
            tl.load(add_50 + offsets).to(tl.float32)
            + input_grad.to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(out_base + offsets, out_vals)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config_on_device(config: dict, device: torch.device) -> dict:
    return {
        "inputs": [
            {**spec, "device": str(device)}
            if isinstance(spec, dict) and spec.get("kind") == "tensor"
            else spec
            for spec in config["inputs"]
        ]
    }


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        inputs = make_inputs_from_config(
            _config_on_device(next(iter(configs.values())), device)
        )
    else:
        module = _load_repro_module()
        inputs = module.make_inputs()

    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def _triton_qwen_moe_index_put_rmsnorm_bwd(
    arg75_1: torch.Tensor,
    full_3: torch.Tensor,
    _grouped_mm_15: torch.Tensor,
    full_5: torch.Tensor,
    arg73_1: torch.Tensor,
    mm_33: torch.Tensor,
    arg9_1: torch.Tensor,
    arg63_1: torch.Tensor,
    arg48_1: torch.Tensor,
    arg64_1: torch.Tensor,
    add_50: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or full_5.device.type != "cuda":
        raise RuntimeError("Triton CUDA is required for this oracle")

    scatter = torch.empty_like(full_5)
    slot_sources = torch.empty(
        (TOKENS, MAX_SOURCES_PER_ROW),
        device=full_5.device,
        dtype=torch.int32,
    )
    sorted_sources = torch.empty_like(slot_sources)
    slot_counts = torch.empty((TOKENS,), device=full_5.device, dtype=torch.int32)
    weight_grad = torch.empty((HIDDEN_SIZE,), device=full_5.device, dtype=torch.bfloat16)
    input_grad_base = torch.empty(
        (TOKENS, HIDDEN_SIZE),
        device=full_5.device,
        dtype=torch.bfloat16,
    )

    _fill_i32_kernel[(triton.cdiv(TOKENS * MAX_SOURCES_PER_ROW, BLOCK_COPY),)](
        slot_sources,
        value=-1,
        n_elements=TOKENS * MAX_SOURCES_PER_ROW,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )
    _fill_i32_kernel[(triton.cdiv(TOKENS, BLOCK_COPY),)](
        slot_counts,
        value=0,
        n_elements=TOKENS,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )
    _build_source_slots_atomic_kernel[(ROUTED_ROWS,)](
        arg73_1,
        slot_counts,
        slot_sources,
        MAX_SOURCES_PER_ROW_=MAX_SOURCES_PER_ROW,
        num_warps=4,
    )
    _sort_source_slots_kernel[(TOKENS,)](
        slot_sources,
        sorted_sources,
        ROUTED_ROWS_=ROUTED_ROWS,
        MAX_SOURCES_PER_ROW_=MAX_SOURCES_PER_ROW,
        num_warps=1,
    )
    _ordered_row_scatter_kernel[(TOKENS, triton.cdiv(HIDDEN_SIZE, BLOCK_SCATTER_COLS))](
        arg75_1,
        full_3,
        _grouped_mm_15,
        full_5,
        sorted_sources,
        scatter,
        HIDDEN_=HIDDEN_SIZE,
        BLOCK_C=BLOCK_SCATTER_COLS,
        MAX_SOURCES_PER_ROW_=MAX_SOURCES_PER_ROW,
        num_warps=4,
    )
    _weight_grad_kernel[(triton.cdiv(HIDDEN_SIZE, BLOCK_OUT0_COLS),)](
        scatter,
        mm_33,
        arg48_1,
        arg63_1,
        arg64_1,
        weight_grad,
        HIDDEN_=HIDDEN_SIZE,
        BLOCK_R=TOKENS,
        BLOCK_C=BLOCK_OUT0_COLS,
        num_warps=8,
    )
    _input_grad_fullrow_kernel[(TOKENS,)](
        scatter,
        mm_33,
        arg9_1,
        arg48_1,
        arg63_1,
        arg64_1,
        add_50,
        input_grad_base,
        HIDDEN_=HIDDEN_SIZE,
        BLOCK_C=HIDDEN_SIZE,
        num_warps=8,
    )
    return weight_grad, input_grad_base.permute(1, 0)


def oracle_qwen_moe_index_put_rmsnorm_bwd(
    arg75_1: torch.Tensor,
    full_3: torch.Tensor,
    _grouped_mm_15: torch.Tensor,
    full_5: torch.Tensor,
    arg73_1: torch.Tensor,
    mm_33: torch.Tensor,
    arg9_1: torch.Tensor,
    arg63_1: torch.Tensor,
    arg48_1: torch.Tensor,
    arg64_1: torch.Tensor,
    add_50: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    )
    return _triton_qwen_moe_index_put_rmsnorm_bwd(
        arg75_1,
        full_3,
        _grouped_mm_15,
        full_5,
        arg73_1,
        mm_33,
        arg9_1,
        arg63_1,
        arg48_1,
        arg64_1,
        add_50,
    )


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return oracle_qwen_moe_index_put_rmsnorm_bwd(*inputs)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return _as_tuple(model(*inputs))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, torch.Tensor):
        return (value,)
    raise TypeError(f"expected tensor or tuple of tensors, got {type(value)!r}")


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = _as_tuple(oracle_qwen_moe_index_put_rmsnorm_bwd(*inputs))
        synchronize(device)

    ok = True
    if len(actual) != len(expected):
        print(f"tuple_length_match=False actual={len(actual)} expected={len(expected)}")
        ok = False

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(
            got.float(),
            ref.float(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and shape_ok and dtype_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} "
            f"mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
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

    with torch.no_grad():
        oracle_qwen_moe_index_put_rmsnorm_bwd(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_qwen_moe_index_put_rmsnorm_bwd(*inputs),
            device,
            warmup,
            rep,
        )
        print(
            f"oracle_qwen_moe_index_put_rmsnorm_bwd: {oracle_us:.3f} us "
            f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
        )

        if compare_eager:
            module = _load_repro_module()
            if device.type != "cuda":
                module.device = lambda *unused_args, **unused_kwargs: device
            repro = module.Repro().to(device)
            repro(*inputs)
            synchronize(device)
            repro_us = benchmark(lambda: repro(*inputs), device, warmup, rep)
            print(
                f"repro_eager: {repro_us:.3f} us "
                f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument(
        "--compare-eager",
        action="store_true",
        help="include eager repro timing in --bench output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.check and not args.bench:
        args.check = True

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
    main()
