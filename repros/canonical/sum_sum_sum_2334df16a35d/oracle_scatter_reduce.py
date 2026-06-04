"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the BEiT average-pool backward `slice_scatter` as a structured token scatter-reduce, deriving the `[768]` addmm-gradient reduction and `[768]` gamma-style reduction directly from the pre-scatter row gradient while materializing only the required `[768, 25216]` transposed side output, whereas Inductor currently materializes the full `[128, 197, 768]` zero-prefix `slice_scatter` result and schedules the two reductions plus returned permute as generic consumers; Inductor cannot do this today because its scheduler/codegen does not model `unsqueeze`/`expand`/`div` feeding `slice_scatter` as a structured scatter-reduce with a zero prefix, broadcasted patch tokens, materialized side stores, and reducer epilogues; the fix is SCATTER_REDUCE: add a structured average-pool-backward `slice_scatter` lowering that accumulates reductions from the source row-gradient tile while emitting any required side-output stores."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPRO_ID = "sum_sum_sum_2334df16a35d"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_beit_base_patch16_224_train_001_68372476"

N = 128
PATCH_TOKENS = 196
TOKENS_WITH_PREFIX = 197
D = 768
FLAT_TOKENS = N * TOKENS_WITH_PREFIX



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


def layernorm_row_grad(
    mm: torch.Tensor,
    arg110_1: torch.Tensor,
    arg307_1: torch.Tensor,
    arg309_1: torch.Tensor,
) -> torch.Tensor:
    weighted = mm * arg110_1
    row_sum = weighted.sum(dim=1, keepdim=True)
    row_dot = (weighted * arg307_1).sum(dim=1, keepdim=True)
    return arg309_1 * (weighted * D - row_sum - arg307_1 * row_dot)


def oracle_scatter_reduce(
    mm: torch.Tensor,
    arg110_1: torch.Tensor,
    arg307_1: torch.Tensor,
    arg309_1: torch.Tensor,
    arg106_1: torch.Tensor,
    arg306_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_grad = layernorm_row_grad(mm, arg110_1, arg307_1, arg309_1)
    patch_grad = row_grad / PATCH_TOKENS

    addmm_tokens = arg306_1.view(N, TOKENS_WITH_PREFIX, D)
    out_addmm = (addmm_tokens[:, 1:, :] * patch_grad[:, None, :]).sum(dim=(0, 1))

    scaled_for_side = patch_grad * arg106_1
    materialized = torch.empty(
        (N, TOKENS_WITH_PREFIX, D),
        device=row_grad.device,
        dtype=row_grad.dtype,
    )
    materialized[:, 0, :].zero_()
    materialized[:, 1:, :] = scaled_for_side[:, None, :]

    flat = materialized.view(FLAT_TOKENS, D)
    transposed_side = flat.permute(1, 0)
    out_scaled = flat.sum(dim=0)
    return out_addmm, transposed_side, out_scaled


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_scatter_reduce(*inputs)
        synchronize(device)

    ok = True
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


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
    device: torch.device,
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        synchronize(device)
    return compiled


def run_bench(device: torch.device, warmup: int, rep: int, no_compile: bool) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        oracle_scatter_reduce(*inputs)
        synchronize(device)

        oracle_us = benchmark(lambda: oracle_scatter_reduce(*inputs), device, warmup, rep)

    print(f"oracle_scatter_reduce: {oracle_us:.3f} us shape={SHAPE_LABEL} device={device}")

    if no_compile:
        return

    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device

    compile_configs = [
        ("default", {}),
        ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    ]
    for label, config in compile_configs:
        model = module.Repro().to(device)
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config, device)
            compiled_us = benchmark(lambda: compiled(*inputs), device, warmup, rep)
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(
            device=device,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    main()
