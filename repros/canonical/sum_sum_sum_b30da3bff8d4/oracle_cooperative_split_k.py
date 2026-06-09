"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GoogleFnet layer-norm backward, dgamma/dbeta reductions, and select_scatter output with Triton row reductions plus atomic cooperative split-K column reductions, whereas Inductor currently emits separate generic reduction and select_scatter kernels around materialized intermediates; Inductor cannot do this today because its scheduler/codegen cannot fuse LN-backward row reductions with sibling column reductions and full select_scatter side-output stores while cooperatively splitting the reduction across the batch-sequence dimension; the fix is COOPERATIVE_SPLIT_K: teach Inductor to generate a specialized layer-norm-backward schedule that keeps row reductions and select_scatter stores in one output kernel while accumulating dgamma and dbeta with cooperative split-K reductions."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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

REPRO_ID = "sum_sum_sum_b30da3bff8d4"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_GoogleFnet_train_001_a1091d37"

B = 32
S = 512
M = B * S
C = 768
LANES = 2
BLOCK_ROW_C = 1024
ROWS_PER_SPLIT = 4
NUM_ROW_SPLITS = math.ceil(M / ROWS_PER_SPLIT)


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


if triton is not None:

    @triton.jit
    def _fused_ln_select_atomic_split_k_kernel(
        mm_ptr,
        residual_ptr,
        gamma_ptr,
        xhat_ptr,
        scale_ptr,
        full_ptr,
        out_ptr,
        dgamma_ptr,
        dbeta_ptr,
        mm_stride_m: tl.constexpr,
        mm_stride_c: tl.constexpr,
        residual_stride_b: tl.constexpr,
        residual_stride_s: tl.constexpr,
        residual_stride_c: tl.constexpr,
        gamma_stride_c: tl.constexpr,
        xhat_stride_b: tl.constexpr,
        xhat_stride_s: tl.constexpr,
        xhat_stride_c: tl.constexpr,
        scale_stride_b: tl.constexpr,
        scale_stride_s: tl.constexpr,
        scale_stride_c: tl.constexpr,
        full_stride_b: tl.constexpr,
        full_stride_s: tl.constexpr,
        full_stride_c: tl.constexpr,
        full_stride_l: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_s: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_l: tl.constexpr,
        ROWS_PER_SPLIT_: tl.constexpr,
        S_: tl.constexpr,
        M_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_C_)
        c_mask = c_offsets < C_
        gamma = tl.load(gamma_ptr + c_offsets * gamma_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        accum_dgamma = tl.full([BLOCK_C_], 0.0, tl.float32)
        accum_dbeta = tl.full([BLOCK_C_], 0.0, tl.float32)

        for row_offset in tl.static_range(0, ROWS_PER_SPLIT_):
            row = pid * ROWS_PER_SPLIT_ + row_offset
            row_active = row < M_
            b_idx = row // S_
            s_idx = row - b_idx * S_
            mask = c_mask & row_active

            mm_offsets = row * mm_stride_m + c_offsets * mm_stride_c
            residual_offsets = (
                b_idx * residual_stride_b
                + s_idx * residual_stride_s
                + c_offsets * residual_stride_c
            )
            xhat_offsets = b_idx * xhat_stride_b + s_idx * xhat_stride_s + c_offsets * xhat_stride_c
            scale_offset = b_idx * scale_stride_b + s_idx * scale_stride_s
            full_lane1_offsets = (
                b_idx * full_stride_b
                + s_idx * full_stride_s
                + c_offsets * full_stride_c
                + full_stride_l
            )
            out_lane0_offsets = (
                b_idx * out_stride_b
                + s_idx * out_stride_s
                + c_offsets * out_stride_c
            )

            mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
            xhat = tl.load(xhat_ptr + xhat_offsets, mask=mask, other=0.0).to(tl.float32)
            row_scale = tl.load(scale_ptr + scale_offset + 0 * scale_stride_c, mask=row_active, other=0.0).to(tl.float32)

            upstream = mm + residual
            weighted = upstream * gamma
            sum_weighted = tl.sum(weighted, axis=0)
            sum_weighted_xhat = tl.sum(weighted * xhat, axis=0)
            dx = row_scale * (weighted * C_ - sum_weighted - xhat * sum_weighted_xhat)
            lane1 = tl.load(full_ptr + full_lane1_offsets, mask=mask, other=0.0).to(tl.float32)

            lane_offsets = tl.arange(0, 2)
            out_pair_offsets = out_lane0_offsets[:, None] + lane_offsets[None, :] * out_stride_l
            out_pair = tl.where(lane_offsets[None, :] == 0, dx[:, None], lane1[:, None])
            tl.store(out_ptr + out_pair_offsets, out_pair, mask=mask[:, None])
            accum_dbeta += upstream
            accum_dgamma += upstream * xhat

        tl.atomic_add(dbeta_ptr + c_offsets, accum_dbeta, sem="relaxed", mask=c_mask)
        tl.atomic_add(dgamma_ptr + c_offsets, accum_dgamma, sem="relaxed", mask=c_mask)


def oracle_cooperative_split_k(
    mm_50: torch.Tensor,
    mul_312: torch.Tensor,
    arg6_1: torch.Tensor,
    arg60_1: torch.Tensor,
    arg164_1: torch.Tensor,
    full_2: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm_50.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    dgamma = torch.empty((C,), device=mm_50.device, dtype=mm_50.dtype)
    dbeta = torch.empty((C,), device=mm_50.device, dtype=mm_50.dtype)
    select_scatter = torch.empty_strided(
        tuple(full_2.shape),
        tuple(full_2.stride()),
        device=full_2.device,
        dtype=full_2.dtype,
    )
    dgamma.zero_()
    dbeta.zero_()

    _fused_ln_select_atomic_split_k_kernel[(NUM_ROW_SPLITS,)](
        mm_50,
        mul_312,
        arg6_1,
        arg60_1,
        arg164_1,
        full_2,
        select_scatter,
        dgamma,
        dbeta,
        mm_stride_m=mm_50.stride(0),
        mm_stride_c=mm_50.stride(1),
        residual_stride_b=mul_312.stride(0),
        residual_stride_s=mul_312.stride(1),
        residual_stride_c=mul_312.stride(2),
        gamma_stride_c=arg6_1.stride(0),
        xhat_stride_b=arg60_1.stride(0),
        xhat_stride_s=arg60_1.stride(1),
        xhat_stride_c=arg60_1.stride(2),
        scale_stride_b=arg164_1.stride(0),
        scale_stride_s=arg164_1.stride(1),
        scale_stride_c=arg164_1.stride(2),
        full_stride_b=full_2.stride(0),
        full_stride_s=full_2.stride(1),
        full_stride_c=full_2.stride(2),
        full_stride_l=full_2.stride(3),
        out_stride_b=select_scatter.stride(0),
        out_stride_s=select_scatter.stride(1),
        out_stride_c=select_scatter.stride(2),
        out_stride_l=select_scatter.stride(3),
        ROWS_PER_SPLIT_=ROWS_PER_SPLIT,
        S_=S,
        M_=M,
        C_=C,
        BLOCK_C_=BLOCK_ROW_C,
        num_warps=4,
    )
    return dgamma, dbeta, select_scatter


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    return module.Repro().to(device)(*inputs)


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


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([32, 512, 768], f32), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([32, 512, 768, 2], f32), S([32, 512, 768]))")
def oracle_forward(inputs):
    return oracle_cooperative_split_k(*inputs)


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
