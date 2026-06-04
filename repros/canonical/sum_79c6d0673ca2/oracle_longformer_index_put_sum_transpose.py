"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle collapses the Longformer overlapping-chunk index_put, scale, hidden-dimension sum, and returned [768, 2048] transpose into one indexed scatter buffer view reused for the full returned tuple, whereas Inductor lowers the accumulate=True index_put as a generic atomic scatter and then schedules the as_strided/view/permute/div/sum/permute consumers as separate layout and reduction work; Inductor cannot do this today because its scheduler/codegen does not recognize a one-dimensional duplicate-index scatter feeding both a reduction epilogue and a required transposed side-output store through view/as_strided layout aliases; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that can emit the scaled transposed side store and the hidden-dimension sum from the same scatter/load tile."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPRO_ID = "sum_79c6d0673ca2"
SHAPE_LABEL = "torchbench_hf_longformer_train_005_7412e5df"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 2048
HIDDEN = 768
SCALE = 0.125



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
        inputs = _load_repro_module().make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def oracle_index_put_sum_transpose(
    bmm_47: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-pattern oracle for index_put(accumulate=True) -> sum + transpose."""
    updates = bmm_47.reshape(-1)
    scattered = torch.ops.aten.index_put.default(full_15, [view_38], updates, True)

    # The captured as_strided/view/permute chain is equivalent to this row-major
    # [2048, 768] view before the div result is transposed for output[1].
    scaled = scattered.view(ROWS, HIDDEN) * SCALE
    reduced = scaled.sum(dim=0)
    transposed = scaled.permute(1, 0)
    return reduced, transposed


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _load_repro_module().Repro().to(device)
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
        actual = oracle_index_put_sum_transpose(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

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
        oracle_index_put_sum_transpose(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_index_put_sum_transpose(*inputs),
            device,
            warmup,
            rep,
        )

    print(
        f"oracle_index_put_sum_transpose: {oracle_us:.3f} us "
        f"shape={SHAPE_LABEL} device={device}"
    )

    if no_compile:
        return

    module = _load_repro_module()
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
