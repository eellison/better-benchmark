"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DeiT distilled two-token select-scatter layer-norm-backward return tuple by deriving the token-0 and token-1 row reductions directly from the two `[128,768]` sources, scattering only those lanes into the required zero-filled transposed side output, and accumulating all three returned channel reductions from the same sparse producer, whereas Inductor currently materializes the dense `[128,198,768]` zero/select_scatter/add tensor and schedules the row reductions, sibling channel reductions, and permute side output as separate generic work; Inductor cannot do this today because scheduler/codegen does not model zero-fill `select_scatter` as a structured scatter-reduce producer that can feed row-wise layer-norm backward plus full side-output stores and sibling reductions; the fix is SCATTER_REDUCE: add a structured select-scatter lowering that maps sparse token sources directly into row-reduction epilogues, emits required materialized scatter stores, and accumulates compatible channel reductions without materializing the dense producer."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPRO_ID = "sum_sum_sum_33f45f532ce3"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_deit_base_distilled_patch16_224_train_2f2590a2"

BATCH = 128
TOKENS = 198
CHANNELS = 768
ROWS = BATCH * TOKENS

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


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

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _layernorm_lane_grad(
    source: torch.Tensor,
    gamma: torch.Tensor,
    xhat: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    weighted = source * gamma
    row_sum = weighted.sum(dim=1, keepdim=True)
    row_dot = (weighted * xhat).sum(dim=1, keepdim=True)
    return scale * (weighted * CHANNELS - row_sum - xhat * row_dot)


def oracle_structured_select_scatter_reduce(
    mm: torch.Tensor,
    mm_2: torch.Tensor,
    primals_151: torch.Tensor,
    mul_84: torch.Tensor,
    div_2: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1

    token0_xhat = mul_84[:, 0, :]
    token1_xhat = mul_84[:, 1, :]
    token0_scale = div_2[:, 0, :]
    token1_scale = div_2[:, 1, :]

    grad_token0 = _layernorm_lane_grad(mm_2, primals_151, token0_xhat, token0_scale)
    grad_token1 = _layernorm_lane_grad(mm, primals_151, token1_xhat, token1_scale)

    out_mul_xhat = (mm_2 * token0_xhat + mm * token1_xhat).sum(dim=0)
    out_scatter_sum = (mm_2 + mm).sum(dim=0)
    out_grad_sum = (grad_token0 + grad_token1).sum(dim=0)

    materialized = torch.zeros(
        (ROWS, CHANNELS),
        device=grad_token0.device,
        dtype=grad_token0.dtype,
    )
    materialized[0::TOKENS, :] = grad_token0
    materialized[1::TOKENS, :] = grad_token1
    transposed_grad = materialized.t()
    return out_mul_xhat, out_scatter_sum, transposed_grad, out_grad_sum


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


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = _as_tuple(oracle_structured_select_scatter_reduce(*inputs))
        synchronize(device)

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
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


def run_bench(device: torch.device, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)

    logical_read_bytes = (
        2 * BATCH * CHANNELS * 4
        + CHANNELS * 4
        + 2 * BATCH * CHANNELS * 4
        + 2 * BATCH * 4
    )
    logical_write_bytes = (3 * CHANNELS + ROWS * CHANNELS) * 4

    with torch.no_grad():
        oracle_structured_select_scatter_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_structured_select_scatter_reduce(*inputs),
            device,
            warmup,
            rep,
        )

    print(
        f"oracle_structured_select_scatter_reduce: {oracle_us:.3f} us "
        f"impl=torch shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )
    print(f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs and strides to repro.py")
    parser.add_argument("--bench", action="store_true", help="time the full-scope oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    with torch.no_grad():
        if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
            sys.exit(1)
        if args.bench:
            run_bench(device=device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
