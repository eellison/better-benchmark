"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle accumulates the Qwen MoE row-index `index_put(accumulate=True)` into logical `[2048, 2048]` token rows and immediately reuses those rows for the RMSNorm weight-gradient sum plus the returned transposed input-gradient store, whereas Inductor currently materializes the BF16 row-scatter result, adds `mm_23`, then schedules the `[0, 1]` weight-gradient reduction and RMSNorm-backward/permute side output as generic consumers; Inductor cannot do this today because its scheduler/codegen does not recognize a one-dimensional MoE routing-index scatter feeding row-local RMSNorm backward reductions as a structured scatter-reduce with a required layout-changing side store; the fix is SCATTER_REDUCE: add a post-grad row-index scatter-reduce template that buckets or gathers routing rows per token, computes the row-local RMSNorm-backward epilogue once, accumulates the `[2048]` weight gradient, and stores the `[2048, 2048]` transpose directly."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPRO_ID = "sum_sum_c39c7b0d801a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_qwen_qwen3-30b-a3b_001_f65095c5"
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
        config = _config_on_device(next(iter(configs.values())), device)
        inputs = make_inputs_from_config(config)
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
    values = torch.where(mask, fill_value, grouped_mm)
    return torch.ops.aten.index_put.default(initial, [row_index], values, True)


def oracle_moe_index_put_reduce(
    arg107_1: torch.Tensor,
    full_3: torch.Tensor,
    _grouped_mm_11: torch.Tensor,
    full_5: torch.Tensor,
    arg105_1: torch.Tensor,
    mm_23: torch.Tensor,
    arg20_1: torch.Tensor,
    arg95_1: torch.Tensor,
    arg96_1: torch.Tensor,
    add_34: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor]:
    scatter_rows = row_index_accumulate(
        arg107_1, full_3, _grouped_mm_11, full_5, arg105_1
    )
    token_grad_2d = scatter_rows + mm_23
    token_grad = token_grad_2d.view(_shape_param_0)

    residual_f32 = arg95_1.float()
    rstd = arg96_1
    normalized_bf16 = (residual_f32 * rstd).to(torch.bfloat16)

    weight_grad = (token_grad * normalized_bf16).sum(
        dim=(0, 1), keepdim=True
    ).view(_shape_param_1)

    weighted_grad = (token_grad * arg20_1).float()
    row_dot = (weighted_grad * residual_f32).sum(dim=2, keepdim=True)
    rms_correction = row_dot * -0.5 * torch.pow(rstd, 3)
    correction = (rms_correction.expand(_shape_param_2) / HIDDEN_SIZE) * (
        torch.pow(residual_f32, 1.0) * 2.0
    )
    input_grad = weighted_grad * rstd + correction

    transposed_input_grad = (
        add_34 + input_grad.to(torch.bfloat16)
    ).view(_shape_param_3).permute(1, 0)
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


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_moe_index_put_reduce(*inputs)
        synchronize(device)

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == ref.stride()
        dtype_ok = got.dtype == ref.dtype
        ok = ok and value_ok and stride_ok and dtype_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"dtype={got.dtype} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
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
        oracle_moe_index_put_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_moe_index_put_reduce(*inputs),
            device,
            warmup,
            rep,
        )
        print(
            f"oracle_moe_index_put_reduce: {oracle_us:.3f} us "
            f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
        )

        if compare_eager:
            module = _load_repro_module()
            repro = module.Repro().to(device)
            repro(*inputs)
            synchronize(device)
            eager_us = benchmark(lambda: repro(*inputs), device, warmup, rep)
            print(
                f"repro_eager: {eager_us:.3f} us "
                f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--compare-eager",
        action="store_true",
        help="include eager repro timing in --bench output",
    )
    args = parser.parse_args()

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
