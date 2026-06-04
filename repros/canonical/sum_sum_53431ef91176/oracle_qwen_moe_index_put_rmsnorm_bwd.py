"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Qwen3 MoE row-index scatter-add plus residual-add RMSNorm backward tuple, reusing the accumulated `[2048, 2048]` token-gradient rows for both the `[2048]` weight-gradient reduction and the returned transposed input-gradient store, whereas Inductor currently materializes the masked grouped-mm source and generic `index_put(accumulate=True)` result, adds the dense residual branch, then schedules the RMSNorm reductions and layout-changing side store as ordinary consumers; Inductor cannot do this today because its scheduler/codegen does not recognize one-dimensional duplicate routing-index row updates feeding a full RMSNorm-backward tuple as a structured scatter-reduce with a required materialized transpose epilogue; the fix is SCATTER_REDUCE: add a Qwen/MoE row-index scatter-reduce template that accumulates routed rows once, folds the residual add before RMSNorm-backward, emits the per-hidden weight-gradient reduction, and stores the transposed input gradient directly."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPRO_ID = "sum_sum_53431ef91176"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_qwen_qwen3-30b-a3b_001_11dcb1c7"
HIDDEN_SIZE = 2048



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


def row_index_accumulate(
    mask: torch.Tensor,
    fill_value: torch.Tensor,
    grouped_mm: torch.Tensor,
    initial: torch.Tensor,
    row_index: torch.Tensor,
) -> torch.Tensor:
    values = torch.ops.aten.where.self(mask, fill_value, grouped_mm)
    return torch.ops.aten.index_put.default(initial, [row_index], values, True)


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
    scatter_rows = row_index_accumulate(
        arg75_1,
        full_3,
        _grouped_mm_15,
        full_5,
        arg73_1,
    )
    token_grad_2d = torch.ops.aten.add.Tensor(scatter_rows, mm_33)
    token_grad = torch.ops.aten.view.default(token_grad_2d, _shape_param_0)

    residual_2d = torch.ops.aten.view.default(arg63_1, _shape_param_1)
    residual = torch.ops.aten.add.Tensor(arg48_1, residual_2d)
    residual_f32 = torch.ops.prims.convert_element_type.default(
        residual,
        torch.float32,
    )
    normalized_bf16 = torch.ops.prims.convert_element_type.default(
        torch.ops.aten.mul.Tensor(residual_f32, arg64_1),
        torch.bfloat16,
    )

    weight_grad = torch.ops.aten.view.default(
        torch.ops.aten.sum.dim_IntList(
            torch.ops.aten.mul.Tensor(token_grad, normalized_bf16),
            [0, 1],
            True,
        ),
        _shape_param_2,
    )

    weighted_token_grad_bf16 = torch.ops.aten.mul.Tensor(token_grad, arg9_1)
    weighted_token_grad = torch.ops.prims.convert_element_type.default(
        weighted_token_grad_bf16,
        torch.float32,
    )
    row_dot = torch.ops.aten.sum.dim_IntList(
        torch.ops.aten.mul.Tensor(weighted_token_grad, residual_f32),
        [2],
        True,
    )
    correction_scale = torch.ops.aten.mul.Tensor(
        torch.ops.aten.mul.Scalar(row_dot, -0.5),
        torch.ops.aten.pow.Tensor_Scalar(arg64_1, 3),
    )
    correction = torch.ops.aten.mul.Tensor(
        torch.ops.aten.div.Scalar(
            torch.ops.aten.expand.default(correction_scale, _shape_param_3),
            HIDDEN_SIZE,
        ),
        torch.ops.aten.mul.Scalar(
            torch.ops.aten.pow.Tensor_Scalar(residual_f32, 1.0),
            2.0,
        ),
    )
    input_grad = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(weighted_token_grad, arg64_1),
        correction,
    )
    transposed_input_grad = torch.ops.aten.permute.default(
        torch.ops.aten.view.default(
            torch.ops.aten.add.Tensor(
                add_50,
                torch.ops.prims.convert_element_type.default(input_grad, torch.bfloat16),
            ),
            _shape_param_4,
        ),
        [1, 0],
    )
    return weight_grad, transposed_input_grad


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
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
        actual = oracle_qwen_moe_index_put_rmsnorm_bwd(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple_length_match=False actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and dtype_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} "
            f"mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
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
