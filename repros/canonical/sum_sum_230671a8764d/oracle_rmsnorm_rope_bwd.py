"""
Gap diagnosis (classification: SCHEDULER_FUSION): this oracle evaluates the Qwen RoPE-backward epilogue once per `[batch, sequence, head, 128]` row, reuses that producer for the RMSNorm hidden-dimension dot product, writes the transposed `[2048, 2048]` input-gradient side output, and forms the `[128]` weight-gradient sum from the same row-structured value, whereas Inductor currently lowers the slice_scatter/permute/RMSNorm-backward graph as separate generic pointwise and reduction work that materializes the RoPE producer and the RMSNorm gradient before the sibling reductions; Inductor cannot do this today because the scheduler/codegen cannot represent a row-persistent reduction whose producer also has a materialized layout-changing side output and a cross-row sibling reduction over only the rotary head dimension; the fix is SCHEDULER_FUSION: add a fused multi-output RMSNorm-backward/rotary template that keeps the per-row dot product in registers, stores the transposed gradient directly, and emits coordinated partial reductions for the weight gradient.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPRO_ID = "sum_sum_230671a8764d"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

sys.path.insert(0, str(REPO_ROOT))


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

    return tuple(value.to(device=device) if isinstance(value, torch.Tensor) else value for value in inputs)


def structured_oracle(
    getitem_81: torch.Tensor,
    unsqueeze_6: torch.Tensor,
    full_3: torch.Tensor,
    unsqueeze_7: torch.Tensor,
    arg5_1: torch.Tensor,
    arg315_1: torch.Tensor,
    arg316_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5

    batch, heads, seq, dims = getitem_81.shape
    half = dims // 2
    tokens = batch * seq
    hidden = heads * dims

    rotary_product = getitem_81 * unsqueeze_6
    rotary_term = torch.cat(
        (rotary_product[..., half:], -rotary_product[..., :half]),
        dim=-1,
    )
    rotary_grad = full_3 + rotary_term + getitem_81 * unsqueeze_7
    row_grad = rotary_grad.permute(0, 2, 1, 3)

    x = arg315_1.view(batch, seq, heads, dims)
    x_f32 = x.float()
    xhat_bf16 = (x_f32 * arg316_1).to(torch.bfloat16)
    weight_grad = (row_grad * xhat_bf16).sum(dim=(0, 1, 2), keepdim=True).view(dims)

    dy_weight = (row_grad * arg5_1).float()
    dot = (dy_weight * x_f32).sum(dim=3, keepdim=True)
    correction = ((dot * -0.5) * (arg316_1**3)).expand_as(dy_weight) / dims
    input_grad = dy_weight * arg316_1 + correction * (x_f32 * 2.0)
    input_grad_t = (
        input_grad.to(torch.bfloat16)
        .contiguous()
        .view(batch, seq, hidden)
        .view(tokens, hidden)
        .permute(1, 0)
    )
    return weight_grad, input_grad_t


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def max_abs_diff(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> float:
    return max((a.float() - e.float()).abs().max().item() for a, e in zip(actual, expected))


def mean_abs_diff(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> float:
    total = 0.0
    count = 0
    for actual_tensor, expected_tensor in zip(actual, expected):
        diff = (actual_tensor.float() - expected_tensor.float()).abs()
        total += diff.sum().item()
        count += diff.numel()
    return total / count


def allclose(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    rtol: float,
    atol: float,
) -> bool:
    return all(torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol) for a, e in zip(actual, expected))


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    module = _load_repro_module()
    inputs = make_inputs(device)
    repro = module.Repro().to(device=device)

    with torch.no_grad():
        expected = repro(*inputs)
        actual = structured_oracle(*inputs)
        synchronize(device)

    diff = max_abs_diff(actual, expected)
    mean_diff = mean_abs_diff(actual, expected)
    ok = allclose(actual, expected, rtol=rtol, atol=atol)
    stride_ok = all(a.stride() == e.stride() for a, e in zip(actual, expected))
    dtype_ok = all(a.dtype == e.dtype for a, e in zip(actual, expected))
    print(f"correct={ok} max_abs_diff={diff:.6g} mean_abs_diff={mean_diff:.6g}")
    print(f"dtype_match={dtype_ok} stride_match={stride_ok}")
    return ok and stride_ok and dtype_ok


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


def run_bench(device: torch.device, warmup: int, rep: int, include_compile: bool) -> None:
    module = _load_repro_module()
    inputs = make_inputs(device)
    repro = module.Repro().to(device=device)

    with torch.no_grad():
        oracle_us = benchmark(lambda: structured_oracle(*inputs), device, warmup, rep)
        repro_us = benchmark(lambda: repro(*inputs), device, warmup, rep)

    print(f"oracle_eager_us={oracle_us:.3f} device={device} warmup={warmup} rep={rep}")
    print(f"repro_eager_us={repro_us:.3f} device={device} warmup={warmup} rep={rep}")

    if not include_compile:
        return

    compiled_repro = torch.compile(repro)
    compiled_oracle = torch.compile(structured_oracle)
    with torch.no_grad():
        compiled_repro(*inputs)
        compiled_oracle(*inputs)
        synchronize(device)
        compiled_repro_us = benchmark(lambda: compiled_repro(*inputs), device, warmup, rep)
        compiled_oracle_us = benchmark(lambda: compiled_oracle(*inputs), device, warmup, rep)

    print(f"repro_compiled_us={compiled_repro_us:.3f} device={device} warmup={warmup} rep={rep}")
    print(f"oracle_compiled_us={compiled_oracle_us:.3f} device={device} warmup={warmup} rep={rep}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--include-compile", action="store_true", help="also benchmark torch.compile variants")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    device = torch.device(args.device)
    if args.check and not run_check(device, args.rtol, args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device, args.warmup, args.rep, args.include_compile)


if __name__ == "__main__":
    main()
