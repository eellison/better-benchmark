"""Full-scope Triton oracle for sum_50185c527fec.

Gap diagnosis (classification: BANDWIDTH_BOUND): the repro is the Swin
attention backward row update

    out[b, h, i, j] = div[b, h, i, j] * (bmm[b, h, i, j]
                                        - sum_k bmm[b, h, i, k] * div[b, h, i, k])

with the output reshaped to contiguous f32[32768, 49, 49]. This oracle consumes
the same bmm tensor, the same padded-stride div tensor, and the same shape
parameters as repro.py, then computes the full multiply, row reduction,
broadcast FMA, and final contiguous output in one Triton kernel. It does not
time a reduction-only subset.

Inductor already emits one fused persistent-reduction Triton kernel for this
scope, so there is no missing high-level fusion, scatter-reduce, split-K,
algebraic-elimination, or recomputation transformation to unlock. The only
possible improvement is a small fixed-shape tiling/autotune choice for this
49-wide row reduction; if that does not beat both required local compile
configs and the historical best compile time, this oracle is diagnosis-only and
the repro should be closed as bandwidth-bound/already at floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
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



REPRO_ID = "sum_50185c527fec"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8192
HEADS = 4
Q = 49
K = 49
BH = BATCH * HEADS
ROWS = BH * Q
OUT_SHAPE = (BH, Q, K)
DIV_SHAPE = (BATCH, HEADS, Q, K)
BMM_STRIDE = (Q * K, K, 1)
DIV_STRIDE = (HEADS * 2432, 2432, K, 1)
R_BLOCK = triton.next_power_of_2(K)
DEFAULT_ROWS_PER_BLOCK = 16
DEFAULT_ROWS_PER_BLOCK_CANDIDATES = (4, 8, 16, 32)
HISTORICAL_BEST_COMPILE_US = 156.63999319076538
CLASSIFICATION = "BANDWIDTH_BOUND"

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _row_reduce_fma_kernel(
    bmm_ptr,
    div_ptr,
    out_ptr,
    ROWS_: tl.constexpr,
    Q_: tl.constexpr,
    K_: tl.constexpr,
    DIV_BH_STRIDE_: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    R_BLOCK_: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)[:, None]
    cols = tl.arange(0, R_BLOCK_)[None, :]
    row_mask = rows < ROWS_
    col_mask = cols < K_
    mask = row_mask & col_mask

    q_index = rows % Q_
    bh_index = rows // Q_
    bmm_offsets = rows * K_ + cols
    div_offsets = bh_index * DIV_BH_STRIDE_ + q_index * K_ + cols

    bmm = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    div = tl.load(div_ptr + div_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    product = bmm * div
    row_sum = tl.sum(tl.where(col_mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    out = tl.fma(-div, row_sum, product)
    tl.store(out_ptr + bmm_offsets, out, mask=mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) and x.device.type != "cuda" else x for x in module.make_inputs())


def _num_warps(rows_per_block: int) -> int:
    if rows_per_block <= 4:
        return 4
    return 8


def _validate_inputs(
    bmm_137: torch.Tensor,
    div_1: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
) -> None:
    assert tuple(bmm_137.shape) == OUT_SHAPE
    assert tuple(div_1.shape) == DIV_SHAPE
    assert tuple(shape_param_0) == DIV_SHAPE
    assert tuple(shape_param_1) == OUT_SHAPE
    assert bmm_137.dtype == torch.float32
    assert div_1.dtype == torch.float32
    assert bmm_137.stride() == BMM_STRIDE
    assert div_1.stride() == DIV_STRIDE


def _oracle_into(
    bmm_137: torch.Tensor,
    div_1: torch.Tensor,
    out: torch.Tensor,
    rows_per_block: int,
) -> torch.Tensor:
    assert out.shape == OUT_SHAPE
    assert out.dtype == torch.float32
    assert out.stride() == BMM_STRIDE

    grid = (triton.cdiv(ROWS, rows_per_block),)
    _row_reduce_fma_kernel[grid](
        bmm_137,
        div_1,
        out,
        ROWS_=ROWS,
        Q_=Q,
        K_=K,
        DIV_BH_STRIDE_=DIV_STRIDE[1],
        ROWS_PER_BLOCK=rows_per_block,
        R_BLOCK_=R_BLOCK,
        num_warps=_num_warps(rows_per_block),
    )
    return out


def oracle_fused(
    bmm_137: torch.Tensor,
    div_1: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
    rows_per_block: int = DEFAULT_ROWS_PER_BLOCK,
) -> torch.Tensor:
    _validate_inputs(bmm_137, div_1, shape_param_0, shape_param_1)
    out = torch.empty(OUT_SHAPE, device=bmm_137.device, dtype=torch.float32)
    return _oracle_into(bmm_137, div_1, out, rows_per_block)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        out = model(*inputs)
    return (out,) if isinstance(out, torch.Tensor) else tuple(out)


def _capture_graph_or_fallback(fn):
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            return lambda: graph.replay(), "cudagraph"
        except Exception:
            return fn, "direct"


def _time_callable(fn, warmup: int, rep: int) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _parse_rows_per_block_candidates(raw: str | None) -> tuple[int, ...]:
    if raw is None or raw.strip() == "":
        return DEFAULT_ROWS_PER_BLOCK_CANDIDATES
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--rows-per-block-candidates produced no candidates")
    for value in values:
        if value <= 0 or value & (value - 1):
            raise ValueError("rows-per-block candidates must be positive powers of two")
    return values


def oracle_forward(inputs):
    return oracle_fused(*inputs)


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
