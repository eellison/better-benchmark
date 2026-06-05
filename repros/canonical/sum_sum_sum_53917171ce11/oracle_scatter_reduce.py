"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the BEiT average-pool backward `slice_scatter` as a structured token scatter with sibling reductions, deriving the two `[768]` reductions from the pre-scatter row gradient and token sums while writing only the required returned `[768, 25216]` transposed side output, whereas Inductor currently materializes the full `[128, 197, 768]` `slice_scatter` result and schedules the reductions and returned permute as generic consumers; Inductor cannot do this today because its scheduler/codegen does not model `unsqueeze`/`expand`/`div` feeding `slice_scatter` as a structured scatter-reduce with materialized side stores and reducer epilogues; the fix is SCATTER_REDUCE: add a structured average-pool-backward `slice_scatter` lowering that accumulates reductions from the source tile while emitting any required side-output stores."""
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



REPRO_ID = "sum_sum_sum_53917171ce11"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_beit_base_patch16_224_train_68372476"

N = 128
PATCH_TOKENS = 196
TOKENS_WITH_PREFIX = 197
D = 768
FLAT_TOKENS = N * TOKENS_WITH_PREFIX



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
    primals_221: torch.Tensor,
    mul_108: torch.Tensor,
    div: torch.Tensor,
) -> torch.Tensor:
    weighted = mm * primals_221
    row_sum = weighted.sum(dim=1, keepdim=True)
    row_dot = (weighted * mul_108).sum(dim=1, keepdim=True)
    return div * (weighted * D - row_sum - mul_108 * row_dot)


def oracle_scatter_reduce(
    mm: torch.Tensor,
    primals_221: torch.Tensor,
    mul_108: torch.Tensor,
    div: torch.Tensor,
    primals_214: torch.Tensor,
    addmm_47: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Structured source-space version of the captured repro outputs."""
    row_grad = layernorm_row_grad(mm, primals_221, mul_108, div)
    patch_grad = row_grad / PATCH_TOKENS

    addmm_tokens = addmm_47.reshape(N, TOKENS_WITH_PREFIX, D)
    out_addmm = (addmm_tokens[:, 1:, :] * patch_grad[:, None, :]).sum(dim=(0, 1))

    scaled_for_side = patch_grad * primals_214
    materialized = torch.empty(
        (N, TOKENS_WITH_PREFIX, D),
        device=row_grad.device,
        dtype=row_grad.dtype,
    )
    materialized[:, 0, :].zero_()
    materialized[:, 1:, :] = scaled_for_side[:, None, :]

    transposed_side = materialized.reshape(FLAT_TOKENS, D).permute(1, 0)
    out_scaled = materialized.reshape(FLAT_TOKENS, D).sum(dim=0)
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
    return oracle_scatter_reduce(*inputs)


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
