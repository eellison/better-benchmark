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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_33f45f532ce3"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_deit_base_distilled_patch16_224_train_2f2590a2"

BATCH = 128
TOKENS = 198
CHANNELS = 768
ROWS = BATCH * TOKENS



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


def oracle_forward(inputs):
    return oracle_structured_select_scatter_reduce(*inputs)


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
