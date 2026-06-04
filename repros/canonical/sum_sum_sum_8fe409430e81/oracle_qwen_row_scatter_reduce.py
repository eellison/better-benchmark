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


REPRO_ID = "sum_sum_sum_8fe409430e81"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_qwen_qwen3-30b-a3b_001_51d90f83"
HIDDEN_SIZE = 2048
TOKENS = 2048
EXPERTS_PER_TOKEN = 8
SCATTER_ROWS = TOKENS * EXPERTS_PER_TOKEN
ROUTER_DIM = 128
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
    def _scatter_token_grad_kernel(
        mm27,
        mm29,
        mm31,
        arg13,
        arg81,
        arg82,
        add40,
        row_dot,
        arg80,
        scatter,
        HIDDEN_: tl.constexpr,
        EXPERTS_PER_TOKEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        src_row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        src_offsets = src_row * HIDDEN_ + cols

        mm_sum = _bf16_add3(
            tl.load(mm27 + src_offsets, mask=col_mask, other=0.0).to(tl.float32),
            tl.load(mm29 + src_offsets, mask=col_mask, other=0.0).to(tl.float32),
            tl.load(mm31 + src_offsets, mask=col_mask, other=0.0).to(tl.float32),
        )
        gamma = tl.load(arg13 + cols, mask=col_mask, other=0.0).to(tl.float32)
        norm_in = tl.load(arg81 + src_offsets, mask=col_mask, other=0.0).to(tl.float32)
        inv_rms = tl.load(arg82 + src_row).to(tl.float32)
        dot = tl.load(row_dot + src_row).to(tl.float32)

        weighted = (mm_sum * gamma).to(tl.bfloat16).to(tl.float32)
        correction = (-0.5 * dot * inv_rms * inv_rms * inv_rms / HIDDEN_) * (norm_in * 2.0)
        input_grad = weighted * inv_rms + correction
        token_grad = (
            tl.load(add40 + src_offsets, mask=col_mask, other=0.0).to(tl.float32)
            + input_grad.to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)

        expert_offsets = tl.arange(0, EXPERTS_PER_TOKEN_)
        dest_rows = tl.load(arg80 + src_row * EXPERTS_PER_TOKEN_ + expert_offsets).to(tl.int64)
        dest_offsets = dest_rows[:, None] * HIDDEN_ + cols[None, :]
        tl.atomic_add(
            scatter + dest_offsets,
            token_grad[None, :],
            sem="relaxed",
            mask=col_mask[None, :],
        )

    @triton.jit
    def _output1_and_row_reduce_kernel(
        scatter,
        mask_ptr,
        full3,
        arg79,
        arg71,
        arg72,
        out1_base,
        row_reduce,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        offsets = row * HIDDEN_ + cols
        masked = tl.load(mask_ptr + row).to(tl.int1)
        full_value = tl.load(full3).to(tl.float32)
        scatter_vals = tl.load(scatter + offsets).to(tl.float32)
        where_vals = tl.where(masked, full_value, scatter_vals)

        route_index = tl.load(arg72 + row).to(tl.int64)
        route_weight = tl.load(arg71 + route_index).to(tl.float32).to(tl.bfloat16).to(tl.float32)
        out1_vals = (where_vals * route_weight).to(tl.bfloat16)
        tl.store(out1_base + offsets, out1_vals)

        rhs = tl.load(arg79 + offsets).to(tl.float32)
        product = (where_vals * rhs).to(tl.bfloat16).to(tl.float32)
        tl.store(row_reduce + row, tl.sum(product, axis=0))

    @triton.jit
    def _route_add_kernel(
        row_reduce,
        arg72,
        route,
        SCATTER_ROWS_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < SCATTER_ROWS_
        values = tl.load(row_reduce + offsets, mask=mask, other=0.0)
        dest = tl.load(arg72 + offsets, mask=mask, other=0).to(tl.int64)
        tl.atomic_add(route + dest, values, sem="relaxed", mask=mask)

    @triton.jit
    def _router_scatter_softmax_backward_kernel(
        router_grad,
        arg71,
        arg70,
        full6,
        arg69,
        arg66,
        arg67,
        arg68,
        out_base,
        EXPERTS_PER_TOKEN_: tl.constexpr,
        ROUTER_DIM_: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0)
        experts = tl.arange(0, BLOCK_E)
        k_offsets = tl.arange(0, EXPERTS_PER_TOKEN_)
        route_offsets = row * EXPERTS_PER_TOKEN_ + k_offsets

        routed = tl.load(router_grad + route_offsets).to(tl.float32)
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


def _triton_router_softmax_backward(
    router_grad: torch.Tensor,
    arg71_1: torch.Tensor,
    arg70_1: torch.Tensor,
    full_6: torch.Tensor,
    arg69_1: torch.Tensor,
    arg66_1: torch.Tensor,
    arg67_1: torch.Tensor,
    arg68_1: torch.Tensor,
) -> torch.Tensor:
    if triton is None or router_grad.device.type != "cuda":
        raise RuntimeError("Triton CUDA is required for this oracle")

    out_base = torch.empty(
        (TOKENS, ROUTER_DIM),
        device=router_grad.device,
        dtype=torch.bfloat16,
    )
    _router_scatter_softmax_backward_kernel[(TOKENS,)](
        router_grad,
        arg71_1,
        arg70_1,
        full_6,
        arg69_1,
        arg66_1,
        arg67_1,
        arg68_1,
        out_base,
        EXPERTS_PER_TOKEN_=EXPERTS_PER_TOKEN,
        ROUTER_DIM_=ROUTER_DIM,
        BLOCK_E=ROUTER_DIM,
        num_warps=4,
    )
    return out_base.permute(1, 0)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or mm_27.device.type != "cuda":
        raise RuntimeError("Triton CUDA is required for this oracle")

    weight_grad = torch.empty((HIDDEN_SIZE,), device=mm_27.device, dtype=torch.bfloat16)
    row_dot = torch.empty((TOKENS,), device=mm_27.device, dtype=torch.float32)
    row_scatter = torch.empty(
        (SCATTER_ROWS, HIDDEN_SIZE),
        device=mm_27.device,
        dtype=torch.bfloat16,
    )
    out1_base = torch.empty(
        (SCATTER_ROWS, HIDDEN_SIZE),
        device=mm_27.device,
        dtype=torch.bfloat16,
    )
    row_reduce = torch.empty((SCATTER_ROWS,), device=mm_27.device, dtype=torch.bfloat16)
    router_grad = torch.empty((SCATTER_ROWS,), device=mm_27.device, dtype=torch.bfloat16)

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
    _copy_kernel[(triton.cdiv(SCATTER_ROWS * HIDDEN_SIZE, BLOCK_COPY),)](
        full_2,
        row_scatter,
        n_elements=SCATTER_ROWS * HIDDEN_SIZE,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )
    _scatter_token_grad_kernel[(TOKENS, triton.cdiv(HIDDEN_SIZE, BLOCK_SCATTER_COLS))](
        mm_27,
        mm_29,
        mm_31,
        arg13_1,
        arg81_1,
        arg82_1,
        add_40,
        row_dot,
        arg80_1,
        row_scatter,
        HIDDEN_=HIDDEN_SIZE,
        EXPERTS_PER_TOKEN_=EXPERTS_PER_TOKEN,
        BLOCK_C=BLOCK_SCATTER_COLS,
        num_warps=4,
    )
    _output1_and_row_reduce_kernel[(SCATTER_ROWS,)](
        row_scatter,
        arg75_1,
        full_3,
        arg79_1,
        arg71_1,
        arg72_1,
        out1_base,
        row_reduce,
        HIDDEN_=HIDDEN_SIZE,
        BLOCK_C=HIDDEN_SIZE,
        num_warps=8,
    )
    _copy_kernel[(triton.cdiv(SCATTER_ROWS, BLOCK_ROUTE),)](
        full_4,
        router_grad,
        n_elements=SCATTER_ROWS,
        BLOCK=BLOCK_ROUTE,
        num_warps=4,
    )
    _route_add_kernel[(triton.cdiv(SCATTER_ROWS, BLOCK_ROUTE),)](
        row_reduce,
        arg72_1,
        router_grad,
        SCATTER_ROWS_=SCATTER_ROWS,
        BLOCK=BLOCK_ROUTE,
        num_warps=4,
    )
    return weight_grad, out1_base.permute(1, 0), router_grad


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
    weight_grad, transposed_row_grad, router_grad = _triton_rmsnorm_scatter_outputs(
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

    transposed_softmax_grad = _triton_router_softmax_backward(
        router_grad,
        arg71_1,
        arg70_1,
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
        actual = _as_tuple(oracle_qwen_row_scatter_reduce(*inputs))
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


def run_bench(device: torch.device, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        oracle_qwen_row_scatter_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_qwen_row_scatter_reduce(*inputs),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_qwen_row_scatter_reduce: {oracle_us:.3f} us "
        f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare full Repro.forward outputs against the oracle")
    parser.add_argument("--bench", action="store_true", help="benchmark the oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
