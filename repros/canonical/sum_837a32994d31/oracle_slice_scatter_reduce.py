"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle writes the returned zero-padded `slice_scatter` tensor and accumulates the sibling masked channel sum from the same source-tile pass, whereas Inductor currently treats the `full -> slice_scatter` side output and the `where(...).sum([0, 2])` reduction as separate generic producers and consumers over `getitem_27`; Inductor cannot do this today because its scheduler/codegen does not model zero-fill `slice_scatter` with a required materialized side output plus a source-space masked reduction epilogue as one structured scatter-reduce template; the fix is SCATTER_REDUCE: add a structured `slice_scatter` lowering that emits the padded side-output stores while accumulating sibling reductions directly from the scattered source tile."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps CPU-only syntax checks working.
    triton = None
    tl = None



from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_837a32994d31"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_demucs_train_003_8b355ac6"

N = 64
C = 1024
W = 364
PAD = 4
PADDED_W = 372
BLOCK_SIZE = 1024
N_TILES = math.ceil((N * PADDED_W) / BLOCK_SIZE)
FINAL_BLOCK = 32



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


def oracle_torch(
    getitem_27: torch.Tensor,
    arg34_1: torch.Tensor,
    full_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.empty((N, C, PADDED_W), device=getitem_27.device, dtype=getitem_27.dtype)
    padded[:, :, :PAD].zero_()
    padded[:, :, PAD : PAD + W] = getitem_27
    padded[:, :, PAD + W :].zero_()
    reduced = torch.where(arg34_1, full_1, getitem_27).sum(dim=(0, 2))
    return padded, reduced


def oracle_triton(
    getitem_27: torch.Tensor,
    arg34_1: torch.Tensor,
    full_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem_27.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    padded = torch.empty((N, C, PADDED_W), device=getitem_27.device, dtype=getitem_27.dtype)
    partial = torch.empty((C, N_TILES), device=getitem_27.device, dtype=torch.float32)
    reduced = torch.empty((C,), device=getitem_27.device, dtype=torch.float32)

    _slice_scatter_reduce_kernel[(C, N_TILES)](
        getitem_27,
        arg34_1,
        full_1,
        padded,
        partial,
        N_=N,
        C_=C,
        W_=W,
        PAD_=PAD,
        PADDED_W_=PADDED_W,
        BLOCK_SIZE_=BLOCK_SIZE,
        N_TILES_=N_TILES,
        num_warps=8,
    )
    _finalize_sum_kernel[(C,)](
        partial,
        reduced,
        N_TILES_=N_TILES,
        FINAL_BLOCK_=FINAL_BLOCK,
        num_warps=1,
    )
    return padded, reduced


def oracle_slice_scatter_reduce(
    getitem_27: torch.Tensor,
    arg34_1: torch.Tensor,
    full_1: torch.Tensor,
    *,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    if impl == "auto":
        impl = "triton" if getitem_27.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(getitem_27, arg34_1, full_1)
    if impl == "torch":
        return oracle_torch(getitem_27, arg34_1, full_1)
    raise ValueError(f"unknown impl: {impl}")


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
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


@oracle_impl(hardware="H100", shapes="(T([64, 1024, 364], f32), T([64, 1024, 364], b8), T([], f32))")
def oracle_forward(inputs):
    return oracle_triton(*inputs)


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
