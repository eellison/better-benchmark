"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Qwen3 structured row-scatter backward tuple, including the RMSNorm weight-gradient reduction, duplicate-index row scatter-add, masked row reduction, router scatter, softmax-backward transpose, and both materialized transposed side outputs, whereas Inductor currently materializes the repeated row expansion, generic index_put buffers, where mask result, sibling row reductions, scatter.src result, and layout-changing permutes as separate generic kernels; Inductor cannot do this today because scheduler/codegen does not model a one-dimensional duplicate row scatter-reduce feeding both a transposed side output and a downstream router softmax-backward scatter as one structured pattern; the fix is SCATTER_REDUCE: add a row-index scatter-reduce lowering that accumulates duplicate token rows once, fuses the masked row reductions and router scatter epilogue, and emits all three Repro.forward outputs with their target layouts."""
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



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_sum_8fe409430e81"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_qwen_qwen3-30b-a3b_001_51d90f83"
HIDDEN_SIZE = 2048
TOKENS = 2048
EXPERTS_PER_TOKEN = 8
SCATTER_ROWS = TOKENS * EXPERTS_PER_TOKEN
ROUTER_DIM = 128
MAX_ROW_SOURCES_PER_ROW = 7
MAX_ROUTE_SOURCES_PER_ROW = 7
BLOCK_OUT0_COLS = 8
BLOCK_SCATTER_COLS = 64
BLOCK_COPY = 1024
BLOCK_ROUTE = 256



if triton is not None:

    @triton.jit
    def _bf16_add3(mm27_vals, mm29_vals, mm31_vals):
        add01 = (mm27_vals + mm29_vals).to(tl.bfloat16).to(tl.float32)
        return (add01 + mm31_vals).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _row_dot_kernel(
        mm27,
        mm29,
        mm31,
        arg13,
        arg81,
        row_dot,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        offsets = row * HIDDEN_ + cols
        mm_sum = _bf16_add3(
            tl.load(mm27 + offsets).to(tl.float32),
            tl.load(mm29 + offsets).to(tl.float32),
            tl.load(mm31 + offsets).to(tl.float32),
        )
        weighted = (mm_sum * tl.load(arg13 + cols).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        norm_in = tl.load(arg81 + offsets).to(tl.float32)
        tl.store(row_dot + row, tl.sum(weighted * norm_in, axis=0))

    @triton.jit
    def _weight_grad_kernel(
        mm27,
        mm29,
        mm31,
        arg81,
        arg82,
        out0,
        HIDDEN_: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols[None, :] < HIDDEN_
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]
        mm_sum = _bf16_add3(
            tl.load(mm27 + offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(mm29 + offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(mm31 + offsets, mask=mask, other=0.0).to(tl.float32),
        )
        norm = (
            tl.load(arg81 + offsets, mask=mask, other=0.0).to(tl.float32)
            * tl.load(arg82 + rows, mask=rows < BLOCK_R, other=0.0).to(tl.float32)[:, None]
        ).to(tl.bfloat16).to(tl.float32)
        product = (mm_sum * norm).to(tl.bfloat16).to(tl.float32)
        tl.store(out0 + cols, tl.sum(product, axis=0), mask=cols < HIDDEN_)

    @triton.jit
    def _copy_kernel(src, dst, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(src + offsets, mask=mask, other=0.0)
        tl.store(dst + offsets, values, mask=mask)

    @triton.jit
    def _fill_i32_kernel(dst, value: tl.constexpr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        tl.store(dst + offsets, value, mask=mask)

    @triton.jit
    def _token_grad_kernel(
        mm27,
        mm29,
        mm31,
        arg13,
        arg81,
        arg82,
        add40,
        row_dot,
        token_grad_out,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        token = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        offsets = token * HIDDEN_ + cols

        mm_sum = _bf16_add3(
            tl.load(mm27 + offsets, mask=col_mask, other=0.0).to(tl.float32),
            tl.load(mm29 + offsets, mask=col_mask, other=0.0).to(tl.float32),
            tl.load(mm31 + offsets, mask=col_mask, other=0.0).to(tl.float32),
        )
        gamma = tl.load(arg13 + cols, mask=col_mask, other=0.0).to(tl.float32)
        norm_in = tl.load(arg81 + offsets, mask=col_mask, other=0.0).to(tl.float32)
        inv_rms = tl.load(arg82 + token).to(tl.float32)
        dot = tl.load(row_dot + token).to(tl.float32)

        weighted = (mm_sum * gamma).to(tl.bfloat16).to(tl.float32)
        correction = (-0.5 * dot * inv_rms * inv_rms * inv_rms / HIDDEN_) * (norm_in * 2.0)
        input_grad = weighted * inv_rms + correction
        token_grad = (
            tl.load(add40 + offsets, mask=col_mask, other=0.0).to(tl.float32)
            + input_grad.to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(token_grad_out + offsets, token_grad, mask=col_mask)

    @triton.jit
    def _output1_and_row_reduce_from_slots_kernel(
        full2,
        token_grad,
        row_slot_sources,
        mask_ptr,
        full3,
        arg79,
        arg71,
        arg72,
        out1_base,
        row_reduce,
        HIDDEN_: tl.constexpr,
        EXPERTS_PER_TOKEN_: tl.constexpr,
        MAX_ROW_SOURCES_PER_ROW_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        offsets = row * HIDDEN_ + cols
        scatter_vals = tl.load(full2 + offsets).to(tl.float32)
        for slot in tl.static_range(0, MAX_ROW_SOURCES_PER_ROW_):
            src = tl.load(
                row_slot_sources + row * MAX_ROW_SOURCES_PER_ROW_ + slot
            ).to(tl.int32)
            valid = src >= 0
            token = tl.where(valid, src // EXPERTS_PER_TOKEN_, 0)
            values = tl.load(
                token_grad + token * HIDDEN_ + cols,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            scatter_vals = tl.where(
                valid,
                (scatter_vals + values).to(tl.bfloat16).to(tl.float32),
                scatter_vals,
            )

        masked = tl.load(mask_ptr + row).to(tl.int1)
        full_value = tl.load(full3).to(tl.float32)
        where_vals = tl.where(masked, full_value, scatter_vals)

        route_index = tl.load(arg72 + row).to(tl.int64)
        route_weight = tl.load(arg71 + route_index).to(tl.float32).to(tl.bfloat16).to(tl.float32)
        out1_vals = (where_vals * route_weight).to(tl.bfloat16)
        tl.store(out1_base + offsets, out1_vals)

        rhs = tl.load(arg79 + offsets).to(tl.float32)
        product = (where_vals * rhs).to(tl.bfloat16).to(tl.float32)
        tl.store(row_reduce + row, tl.sum(product, axis=0))

    @triton.jit
    def _build_route_source_slots_kernel(
        arg72,
        slot_counts,
        slot_sources,
        MAX_ROUTE_SOURCES_PER_ROW_: tl.constexpr,
    ):
        src = tl.program_id(0)
        dest = tl.load(arg72 + src).to(tl.int64)
        slot = tl.atomic_add(slot_counts + dest, 1, sem="relaxed")
        tl.store(
            slot_sources + dest * MAX_ROUTE_SOURCES_PER_ROW_ + slot,
            src,
            mask=slot < MAX_ROUTE_SOURCES_PER_ROW_,
        )

    @triton.jit
    def _sort_route_source_slots_kernel(
        slot_sources,
        sorted_sources,
        SCATTER_ROWS_: tl.constexpr,
        MAX_ROUTE_SOURCES_PER_ROW_: tl.constexpr,
    ):
        dest = tl.program_id(0)
        last = tl.full((), -1, tl.int32)
        for out_slot in tl.static_range(0, MAX_ROUTE_SOURCES_PER_ROW_):
            best = tl.full((), SCATTER_ROWS_, tl.int32)
            for in_slot in tl.static_range(0, MAX_ROUTE_SOURCES_PER_ROW_):
                candidate = tl.load(
                    slot_sources + dest * MAX_ROUTE_SOURCES_PER_ROW_ + in_slot
                ).to(tl.int32)
                take = (candidate >= 0) & (candidate > last) & (candidate < best)
                best = tl.where(take, candidate, best)
            valid = best < SCATTER_ROWS_
            tl.store(
                sorted_sources + dest * MAX_ROUTE_SOURCES_PER_ROW_ + out_slot,
                tl.where(valid, best, -1),
            )
            last = best

    @triton.jit
    def _router_scatter_softmax_backward_from_slots_kernel(
        row_reduce,
        route_slot_sources,
        arg71,
        arg70,
        full4,
        full6,
        arg69,
        arg66,
        arg67,
        arg68,
        out_base,
        EXPERTS_PER_TOKEN_: tl.constexpr,
        ROUTER_DIM_: tl.constexpr,
        MAX_ROUTE_SOURCES_PER_ROW_: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0)
        experts = tl.arange(0, BLOCK_E)
        k_offsets = tl.arange(0, EXPERTS_PER_TOKEN_)
        route_offsets = row * EXPERTS_PER_TOKEN_ + k_offsets

        routed = tl.full((EXPERTS_PER_TOKEN_,), 0.0, tl.float32)
        for k in tl.static_range(0, EXPERTS_PER_TOKEN_):
            route_offset = row * EXPERTS_PER_TOKEN_ + k
            accum = tl.load(full4 + route_offset).to(tl.float32)
            for slot in tl.static_range(0, MAX_ROUTE_SOURCES_PER_ROW_):
                src = tl.load(
                    route_slot_sources + route_offset * MAX_ROUTE_SOURCES_PER_ROW_ + slot
                ).to(tl.int32)
                valid = src >= 0
                value = tl.load(row_reduce + src, mask=valid, other=0.0).to(tl.float32)
                accum = tl.where(valid, accum + value, accum)
            routed = tl.where(k_offsets == k, accum.to(tl.bfloat16).to(tl.float32), routed)

        denom = tl.load(arg70 + row).to(tl.float32)
        route_weights = tl.load(arg71 + route_offsets).to(tl.float32)
        row_sum_correction = -tl.sum(routed * route_weights / denom, axis=0)
        add4 = routed / denom + row_sum_correction

        scatter_vals = tl.load(full6 + row * ROUTER_DIM_ + experts).to(tl.float32)
        seen = experts < 0
        for k in tl.static_range(0, EXPERTS_PER_TOKEN_):
            dest = tl.load(arg69 + row * EXPERTS_PER_TOKEN_ + k).to(tl.int64)
            add4_k = tl.sum(tl.where(k_offsets == k, add4, 0.0), axis=0)
            match = (experts == dest) & (~seen)
            scatter_vals = tl.where(match, add4_k, scatter_vals)
            seen = seen | (experts == dest)

        logits = tl.load(arg66 + row * ROUTER_DIM_ + experts).to(tl.float32)
        max_logit = tl.load(arg67 + row).to(tl.float32)
        normalizer = tl.load(arg68 + row).to(tl.float32)
        softmax_prob = tl.exp(logits - max_logit) / normalizer
        grad_times_prob = scatter_vals * softmax_prob
        row_sum = tl.sum(grad_times_prob, axis=0)
        out = grad_times_prob - softmax_prob * row_sum
        tl.store(out_base + row * ROUTER_DIM_ + experts, out)


def _triton_router_softmax_backward_from_slots(
    row_reduce: torch.Tensor,
    route_slot_sources: torch.Tensor,
    arg71_1: torch.Tensor,
    arg70_1: torch.Tensor,
    full_4: torch.Tensor,
    full_6: torch.Tensor,
    arg69_1: torch.Tensor,
    arg66_1: torch.Tensor,
    arg67_1: torch.Tensor,
    arg68_1: torch.Tensor,
) -> torch.Tensor:
    if triton is None or row_reduce.device.type != "cuda":
        raise RuntimeError("Triton CUDA is required for this oracle")

    out_base = torch.empty(
        (TOKENS, ROUTER_DIM),
        device=row_reduce.device,
        dtype=torch.bfloat16,
    )
    _router_scatter_softmax_backward_from_slots_kernel[(TOKENS,)](
        row_reduce,
        route_slot_sources,
        arg71_1,
        arg70_1,
        full_4,
        full_6,
        arg69_1,
        arg66_1,
        arg67_1,
        arg68_1,
        out_base,
        EXPERTS_PER_TOKEN_=EXPERTS_PER_TOKEN,
        ROUTER_DIM_=ROUTER_DIM,
        MAX_ROUTE_SOURCES_PER_ROW_=MAX_ROUTE_SOURCES_PER_ROW,
        BLOCK_E=ROUTER_DIM,
        num_warps=4,
    )
    return out_base.permute(1, 0)


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

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _triton_rmsnorm_scatter_outputs(
    mm_27: torch.Tensor,
    mm_29: torch.Tensor,
    mm_31: torch.Tensor,
    arg13_1: torch.Tensor,
    arg81_1: torch.Tensor,
    arg82_1: torch.Tensor,
    add_40: torch.Tensor,
    full_2: torch.Tensor,
    arg80_1: torch.Tensor,
    arg75_1: torch.Tensor,
    full_3: torch.Tensor,
    arg79_1: torch.Tensor,
    arg71_1: torch.Tensor,
    arg72_1: torch.Tensor,
    full_4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or mm_27.device.type != "cuda":
        raise RuntimeError("Triton CUDA is required for this oracle")

    weight_grad = torch.empty((HIDDEN_SIZE,), device=mm_27.device, dtype=torch.bfloat16)
    row_dot = torch.empty((TOKENS,), device=mm_27.device, dtype=torch.float32)
    token_grad = torch.empty(
        (TOKENS, HIDDEN_SIZE),
        device=mm_27.device,
        dtype=torch.bfloat16,
    )
    out1_base = torch.empty(
        (SCATTER_ROWS, HIDDEN_SIZE),
        device=mm_27.device,
        dtype=torch.bfloat16,
    )
    row_reduce = torch.empty((SCATTER_ROWS,), device=mm_27.device, dtype=torch.bfloat16)
    row_slot_counts = torch.empty(
        (SCATTER_ROWS,),
        device=mm_27.device,
        dtype=torch.int32,
    )
    row_slot_sources = torch.empty(
        (SCATTER_ROWS, MAX_ROW_SOURCES_PER_ROW),
        device=mm_27.device,
        dtype=torch.int32,
    )
    sorted_row_slot_sources = torch.empty_like(row_slot_sources)
    route_slot_counts = torch.empty(
        (SCATTER_ROWS,),
        device=mm_27.device,
        dtype=torch.int32,
    )
    route_slot_sources = torch.empty(
        (SCATTER_ROWS, MAX_ROUTE_SOURCES_PER_ROW),
        device=mm_27.device,
        dtype=torch.int32,
    )

    _row_dot_kernel[(TOKENS,)](
        mm_27,
        mm_29,
        mm_31,
        arg13_1,
        arg81_1,
        row_dot,
        HIDDEN_=HIDDEN_SIZE,
        BLOCK_C=HIDDEN_SIZE,
        num_warps=8,
    )
    _weight_grad_kernel[(triton.cdiv(HIDDEN_SIZE, BLOCK_OUT0_COLS),)](
        mm_27,
        mm_29,
        mm_31,
        arg81_1,
        arg82_1,
        weight_grad,
        HIDDEN_=HIDDEN_SIZE,
        BLOCK_R=TOKENS,
        BLOCK_C=BLOCK_OUT0_COLS,
        num_warps=8,
    )
    _token_grad_kernel[(TOKENS, triton.cdiv(HIDDEN_SIZE, BLOCK_SCATTER_COLS))](
        mm_27,
        mm_29,
        mm_31,
        arg13_1,
        arg81_1,
        arg82_1,
        add_40,
        row_dot,
        token_grad,
        HIDDEN_=HIDDEN_SIZE,
        BLOCK_C=BLOCK_SCATTER_COLS,
        num_warps=4,
    )
    _fill_i32_kernel[(triton.cdiv(SCATTER_ROWS, BLOCK_ROUTE),)](
        row_slot_counts,
        value=0,
        n_elements=SCATTER_ROWS,
        BLOCK=BLOCK_ROUTE,
        num_warps=4,
    )
    _fill_i32_kernel[
        (triton.cdiv(SCATTER_ROWS * MAX_ROW_SOURCES_PER_ROW, BLOCK_COPY),)
    ](
        row_slot_sources,
        value=-1,
        n_elements=SCATTER_ROWS * MAX_ROW_SOURCES_PER_ROW,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )
    _build_route_source_slots_kernel[(SCATTER_ROWS,)](
        arg80_1,
        row_slot_counts,
        row_slot_sources,
        MAX_ROUTE_SOURCES_PER_ROW_=MAX_ROW_SOURCES_PER_ROW,
        num_warps=4,
    )
    _sort_route_source_slots_kernel[(SCATTER_ROWS,)](
        row_slot_sources,
        sorted_row_slot_sources,
        SCATTER_ROWS_=SCATTER_ROWS,
        MAX_ROUTE_SOURCES_PER_ROW_=MAX_ROW_SOURCES_PER_ROW,
        num_warps=4,
    )
    _output1_and_row_reduce_from_slots_kernel[(SCATTER_ROWS,)](
        full_2,
        token_grad,
        sorted_row_slot_sources,
        arg75_1,
        full_3,
        arg79_1,
        arg71_1,
        arg72_1,
        out1_base,
        row_reduce,
        HIDDEN_=HIDDEN_SIZE,
        EXPERTS_PER_TOKEN_=EXPERTS_PER_TOKEN,
        MAX_ROW_SOURCES_PER_ROW_=MAX_ROW_SOURCES_PER_ROW,
        BLOCK_C=HIDDEN_SIZE,
        num_warps=8,
    )
    _fill_i32_kernel[(triton.cdiv(SCATTER_ROWS, BLOCK_ROUTE),)](
        route_slot_counts,
        value=0,
        n_elements=SCATTER_ROWS,
        BLOCK=BLOCK_ROUTE,
        num_warps=4,
    )
    _fill_i32_kernel[
        (triton.cdiv(SCATTER_ROWS * MAX_ROUTE_SOURCES_PER_ROW, BLOCK_COPY),)
    ](
        route_slot_sources,
        value=-1,
        n_elements=SCATTER_ROWS * MAX_ROUTE_SOURCES_PER_ROW,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )
    _build_route_source_slots_kernel[(SCATTER_ROWS,)](
        arg72_1,
        route_slot_counts,
        route_slot_sources,
        MAX_ROUTE_SOURCES_PER_ROW_=MAX_ROUTE_SOURCES_PER_ROW,
        num_warps=4,
    )
    return weight_grad, out1_base.permute(1, 0), row_reduce, route_slot_sources


def oracle_qwen_row_scatter_reduce(
    mm_27: torch.Tensor,
    mm_29: torch.Tensor,
    mm_31: torch.Tensor,
    arg13_1: torch.Tensor,
    arg81_1: torch.Tensor,
    arg82_1: torch.Tensor,
    add_40: torch.Tensor,
    full_2: torch.Tensor,
    arg80_1: torch.Tensor,
    arg75_1: torch.Tensor,
    full_3: torch.Tensor,
    arg79_1: torch.Tensor,
    arg71_1: torch.Tensor,
    arg72_1: torch.Tensor,
    full_4: torch.Tensor,
    arg70_1: torch.Tensor,
    full_6: torch.Tensor,
    arg69_1: torch.Tensor,
    arg66_1: torch.Tensor,
    arg67_1: torch.Tensor,
    arg68_1: torch.Tensor,
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
    weight_grad, transposed_row_grad, row_reduce, route_slot_sources = (
        _triton_rmsnorm_scatter_outputs(
            mm_27,
            mm_29,
            mm_31,
            arg13_1,
            arg81_1,
            arg82_1,
            add_40,
            full_2,
            arg80_1,
            arg75_1,
            full_3,
            arg79_1,
            arg71_1,
            arg72_1,
            full_4,
        )
    )

    transposed_softmax_grad = _triton_router_softmax_backward_from_slots(
        row_reduce,
        route_slot_sources,
        arg71_1,
        arg70_1,
        full_4,
        full_6,
        arg69_1,
        arg66_1,
        arg67_1,
        arg68_1,
    )
    return weight_grad, transposed_row_grad, transposed_softmax_grad


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


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


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


def oracle_forward(inputs):
    return oracle_qwen_row_scatter_reduce(*inputs)


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
