"""Full-scope Triton oracle for amax_sum_cd5b9c2339af.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the
complete MT5 attention softmax/dropout region from repro.py, including the
view of f32[192,128,128] scores to [32,6,128,128], the f32 position-bias add,
stable last-dimension softmax, Inductor stateless RNG dropout using
tl.rand(seed, flat_offset), dropout scaling, expand/view, and final
non-contiguous [192,128,128] permuted output stride. It differs from Inductor
by using a shape-specialized row-blocked online-softmax/dropout Triton template
that groups eight K=128 rows per program while writing the final permuted
stride directly. Inductor can fuse the graph today, but its generic
persistent-reduction schedule/autotune path does not select this small-K
attention-softmax/dropout row-blocked template under the required configs. The
Inductor change that would fix it is a dedicated or expanded online-softmax
dropout/layout template for K=128 attention rows that includes this row-grouped
schedule and gates it by shape.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "amax_sum_cd5b9c2339af"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 6
Q_LEN = 128
K_LEN = 128
BH = BATCH * HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
ADD_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
OUT_SHAPE = BMM_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, 1, K_LEN)
SEED_INDEX = 79
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
HISTORICAL_BEST_COMPILE_US = 19.36000026762485
CLASSIFICATION = "NEW_PATTERN"

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _softmax_dropout_kernel(
    bmm_ptr,
    add_ptr,
    seeds_ptr,
    out_ptr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_k)
    row_mask = rows < 24576
    col_mask = cols < 128
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * 128 + cols[None, :]

    bmm_vals = tl.load(bmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_vals = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scores = bmm_vals + add_vals
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    numer = tl.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)

    seed = tl.load(seeds_ptr + 79)
    random_vals = tl.rand(seed, offsets.to(tl.uint32))
    keep = (random_vals > 0.1).to(tl.float32)
    out_vals = (numer / denom[:, None]) * keep * 1.1111111111111112

    tl.store(out_ptr + offsets, out_vals, mask=mask)


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _make_inputs(module: Any, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)


def _validate_shape_param(name: str, actual: object, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm_46: torch.Tensor,
    add_70: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
) -> None:
    if not (bmm_46.is_cuda and add_70.is_cuda and inductor_seeds.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_46.dtype != torch.float32 or add_70.dtype != torch.float32:
        raise TypeError(f"expected fp32 score inputs, got {bmm_46.dtype} and {add_70.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 inductor_seeds, got {inductor_seeds.dtype}")
    if tuple(bmm_46.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_46 shape: {tuple(bmm_46.shape)}")
    if tuple(add_70.shape) != ADD_SHAPE:
        raise ValueError(f"unexpected add_70 shape: {tuple(add_70.shape)}")
    if tuple(inductor_seeds.shape) != (84,):
        raise ValueError(f"unexpected inductor_seeds shape: {tuple(inductor_seeds.shape)}")
    if tuple(bmm_46.stride()) != (Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected bmm_46 stride: {tuple(bmm_46.stride())}")
    if tuple(add_70.stride()) != (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected add_70 stride: {tuple(add_70.stride())}")
    _validate_shape_param("_shape_param_0", _shape_param_0, ADD_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, ADD_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _make_output(device: torch.device) -> torch.Tensor:
    return torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=device,
        dtype=torch.float32,
    )


def _launch_oracle(
    bmm_46: torch.Tensor,
    add_70: torch.Tensor,
    inductor_seeds: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    num_warps: int,
) -> torch.Tensor:
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")
    if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
        raise ValueError(f"output must have shape {OUT_SHAPE} and stride {OUT_STRIDE}")
    if out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("output must be CUDA fp32")

    _softmax_dropout_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm_46,
        add_70,
        inductor_seeds,
        out,
        block_m=block_m,
        block_k=K_LEN,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_46: torch.Tensor,
    add_70: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    block_m: int = 8,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(
        bmm_46,
        add_70,
        inductor_seeds,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    out = _make_output(bmm_46.device)
    return _launch_oracle(
        bmm_46,
        add_70,
        inductor_seeds,
        out,
        block_m=block_m,
        num_warps=num_warps,
    )


def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    return oracle_online_softmax(*inputs)


def main() -> None:
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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
