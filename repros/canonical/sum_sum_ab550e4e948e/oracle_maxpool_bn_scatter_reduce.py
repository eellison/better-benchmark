"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the timm VoVNet max-pool-backward `scatter_add` and downstream ReLU/BatchNorm-backward return tuple as a structured scatter-reduce, accumulating the two `[768]` channel summaries while producing the returned `[32,768,14,14]` input-gradient tensor from the same scattered activation-gradient values, whereas Inductor currently materializes the `[24576,196]` pool-backward scatter result and schedules the ReLU mask, two channel reductions, and dependent BatchNorm-backward epilogue as separate generic consumers; Inductor cannot do this today because its scheduler/codegen does not model low-memory max-pool indices feeding a `scatter_add` with sibling channel reductions and a materialized dependent side output as one structured scatter-reduce template; the fix is SCATTER_REDUCE: add a max-pool-backward scatter-reduce lowering that tiles source and destination spatial positions, accumulates channel summaries, and fuses the BatchNorm-backward epilogue stores for the full return tuple."""
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


REPRO_ID = "sum_sum_ab550e4e948e"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_timm_vovnet_train_001_9ab7e113"

N = 32
C = 768
POOL_H = 7
POOL_W = 7
OUT_H = 14
OUT_W = 14
OUT_HW = OUT_H * OUT_W
REDUCTION_SCALE = 1.0 / (N * OUT_H * OUT_W)

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

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def oracle_maxpool_bn_scatter_reduce(
    getitem_18: torch.Tensor,
    getitem_33: torch.Tensor,
    arg180_1: torch.Tensor,
    arg176_1: torch.Tensor,
    arg177_1: torch.Tensor,
    arg178_1: torch.Tensor,
    arg60_1: torch.Tensor,
    arg61_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-tuple source-space oracle for the max-pool scatter plus BN backward."""
    source = getitem_18[:, :C, :, :] + getitem_33
    source_flat = source.reshape(N, C, POOL_H * POOL_W)

    pool_indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg180_1,
        [3, 3],
        [OUT_H, OUT_W],
        [2, 2],
        [0, 0],
        [1, 1],
    ).reshape(N, C, POOL_H * POOL_W)

    scattered_flat = torch.zeros(
        (N, C, OUT_HW),
        device=source.device,
        dtype=source.dtype,
    )
    scattered_flat.scatter_add_(2, pool_indices, source_flat)
    scattered = scattered_flat.view(N, C, OUT_H, OUT_W)

    centered = arg176_1 - arg177_1
    invstd = arg178_1.squeeze((0, 2, 3))
    weight = arg60_1
    bias = arg61_1

    bn_affine = centered * arg178_1
    bn_affine = bn_affine * weight.view(1, C, 1, 1)
    bn_affine = bn_affine + bias.view(1, C, 1, 1)
    relu_mask = bn_affine <= 0
    grad_after_relu = torch.where(relu_mask, full, scattered)

    sum_grad = grad_after_relu.sum(dim=(0, 2, 3))
    sum_grad_centered = (grad_after_relu * centered).sum(dim=(0, 2, 3))

    mean_term = (sum_grad * REDUCTION_SCALE).view(1, C, 1, 1)
    var_term = (
        sum_grad_centered
        * REDUCTION_SCALE
        * invstd
        * invstd
    ).view(1, C, 1, 1)
    affine_scale = (invstd * weight).view(1, C, 1, 1)

    input_grad = (
        grad_after_relu
        - centered * var_term
        - mean_term
    ) * affine_scale
    weight_grad = sum_grad_centered * invstd
    return input_grad, weight_grad


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


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        actual = _as_tuple(oracle_maxpool_bn_scatter_reduce(*inputs))
        expected = reference_outputs(inputs, device)
        synchronize(device)

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} stride_match={stride_ok}"
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
    oracle_fn = lambda: oracle_maxpool_bn_scatter_reduce(*inputs)
    with torch.no_grad():
        oracle_fn()
        synchronize(device)
        oracle_us = benchmark(oracle_fn, device, warmup, rep)
    print(
        f"oracle_maxpool_bn_scatter_reduce: {oracle_us:.3f} us "
        f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the full Repro.forward return tuple against the oracle")
    parser.add_argument("--bench", action="store_true", help="benchmark the oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
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
