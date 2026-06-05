"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete BEiT layer-norm-backward row formula, structured expand/div slice scatter into patch tokens, transposed full side output, and both 768-wide reductions with Triton kernels, whereas Inductor currently lowers the row reductions, zero-fill slice_scatter/expand/div producer, transposed materialized output, and sibling reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not model the zero-fill slice_scatter expand as a scatter-reduce producer that can simultaneously feed reductions and a required full transposed side output; the fix is SCATTER_REDUCE: add a structured slice_scatter/expand-div lowering that streams the repeated patch-token value directly into all reduction consumers while emitting the transposed return tensor."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_sum_2334df16a35d"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_beit_base_patch16_224_train_001_68372476"

N = 128
C = 768
PATCH_TOKENS = 196
TOKENS = 197
Q = N * TOKENS
INV_PATCH_TOKENS = 1.0 / PATCH_TOKENS
ROW_WIDTH_F = float(C)

BLOCK_ROW_C = 1024
BLOCK_Q = 256
BLOCK_C = 32
NUM_Q_TILES = triton.cdiv(Q, BLOCK_Q) if triton is not None else math.ceil(Q / BLOCK_Q)
BLOCK_TILES = 128


def make_inputs(device: torch.device) -> tuple[object, ...]:
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
    def _row_formula_kernel(
        mm_ptr,
        arg110_ptr,
        arg307_ptr,
        arg309_ptr,
        row_value_ptr,
        mm_stride_n: tl.constexpr,
        mm_stride_c: tl.constexpr,
        arg110_stride_c: tl.constexpr,
        arg307_stride_n: tl.constexpr,
        arg307_stride_c: tl.constexpr,
        arg309_stride_n: tl.constexpr,
        row_value_stride_n: tl.constexpr,
        row_value_stride_c: tl.constexpr,
        C_: tl.constexpr,
        ROW_WIDTH_F_: tl.constexpr,
        BLOCK_ROW_C_: tl.constexpr,
    ):
        n_idx = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_ROW_C_)
        mask = c_offsets < C_

        mm = tl.load(
            mm_ptr + n_idx * mm_stride_n + c_offsets * mm_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            arg110_ptr + c_offsets * arg110_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        xhat = tl.load(
            arg307_ptr + n_idx * arg307_stride_n + c_offsets * arg307_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(arg309_ptr + n_idx * arg309_stride_n).to(tl.float32)

        weighted_grad = mm * weight
        row_sum = tl.sum(tl.where(mask, weighted_grad, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, weighted_grad * xhat, 0.0), axis=0)
        row_value = scale * (weighted_grad * ROW_WIDTH_F_ - row_sum - xhat * row_dot)

        tl.store(
            row_value_ptr + n_idx * row_value_stride_n + c_offsets * row_value_stride_c,
            row_value,
            mask=mask,
        )

    @triton.jit
    def _scatter_transpose_reduce_kernel(
        row_value_ptr,
        arg106_ptr,
        arg306_ptr,
        partial_out0_ptr,
        out1_ptr,
        arg106_stride_c: tl.constexpr,
        arg306_stride_q: tl.constexpr,
        arg306_stride_c: tl.constexpr,
        out1_stride_c: tl.constexpr,
        out1_stride_q: tl.constexpr,
        C_: tl.constexpr,
        Q_: tl.constexpr,
        TOKENS_: tl.constexpr,
        INV_PATCH_TOKENS_: tl.constexpr,
        BLOCK_Q_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_q = tl.program_id(1)
        q_offsets = pid_q * BLOCK_Q_ + tl.arange(0, BLOCK_Q_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        q_mask = q_offsets < Q_
        c_mask = c_offsets < C_
        active = q_mask[:, None] & c_mask[None, :]

        n_offsets = q_offsets // TOKENS_
        token_offsets = q_offsets - n_offsets * TOKENS_
        patch_mask = token_offsets > 0

        row_value = tl.load(
            row_value_ptr + n_offsets[:, None] * C_ + c_offsets[None, :],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        arg106 = tl.load(
            arg106_ptr + c_offsets * arg106_stride_c,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        scattered = tl.where(patch_mask[:, None], row_value * INV_PATCH_TOKENS_, 0.0)
        out1 = scattered * arg106[None, :]

        tl.store(
            out1_ptr + c_offsets[None, :] * out1_stride_c + q_offsets[:, None] * out1_stride_q,
            out1,
            mask=active,
        )

        arg306 = tl.load(
            arg306_ptr + q_offsets[:, None] * arg306_stride_q + c_offsets[None, :] * arg306_stride_c,
            mask=active & patch_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        partial_out0 = tl.sum(scattered * arg306, axis=0)
        tl.store(partial_out0_ptr + pid_q * C_ + c_offsets, partial_out0, mask=c_mask)

    @triton.jit
    def _finalize_vectors_kernel(
        partial_out0_ptr,
        row_value_ptr,
        arg106_ptr,
        out0_ptr,
        out2_ptr,
        arg106_stride_c: tl.constexpr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        NUM_Q_TILES_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tile_offsets = tl.arange(0, BLOCK_TILES_)
        c_mask = c_offsets < C_

        partial_mask = (tile_offsets[:, None] < NUM_Q_TILES_) & c_mask[None, :]
        partial = tl.load(
            partial_out0_ptr + tile_offsets[:, None] * C_ + c_offsets[None, :],
            mask=partial_mask,
            other=0.0,
        ).to(tl.float32)
        out0 = tl.sum(partial, axis=0)

        n_mask = (tile_offsets[:, None] < N_) & c_mask[None, :]
        row_value = tl.load(
            row_value_ptr + tile_offsets[:, None] * C_ + c_offsets[None, :],
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        row_sum = tl.sum(row_value, axis=0)
        arg106 = tl.load(
            arg106_ptr + c_offsets * arg106_stride_c,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)

        tl.store(out0_ptr + c_offsets, out0, mask=c_mask)
        tl.store(out2_ptr + c_offsets, row_sum * arg106, mask=c_mask)


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
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    row_value = torch.empty((N, C), device=mm.device, dtype=torch.float32)
    partial_out0 = torch.empty((NUM_Q_TILES, C), device=mm.device, dtype=torch.float32)
    out0 = torch.empty((C,), device=mm.device, dtype=mm.dtype)
    out1 = torch.empty_strided((C, Q), (1, C), device=mm.device, dtype=mm.dtype)
    out2 = torch.empty((C,), device=mm.device, dtype=mm.dtype)

    _row_formula_kernel[(N,)](
        mm,
        arg110_1,
        arg307_1,
        arg309_1,
        row_value,
        mm_stride_n=mm.stride(0),
        mm_stride_c=mm.stride(1),
        arg110_stride_c=arg110_1.stride(0),
        arg307_stride_n=arg307_1.stride(0),
        arg307_stride_c=arg307_1.stride(1),
        arg309_stride_n=arg309_1.stride(0),
        row_value_stride_n=row_value.stride(0),
        row_value_stride_c=row_value.stride(1),
        C_=C,
        ROW_WIDTH_F_=ROW_WIDTH_F,
        BLOCK_ROW_C_=BLOCK_ROW_C,
        num_warps=8,
    )

    _scatter_transpose_reduce_kernel[(triton.cdiv(C, BLOCK_C), NUM_Q_TILES)](
        row_value,
        arg106_1,
        arg306_1,
        partial_out0,
        out1,
        arg106_stride_c=arg106_1.stride(0),
        arg306_stride_q=arg306_1.stride(0),
        arg306_stride_c=arg306_1.stride(1),
        out1_stride_c=out1.stride(0),
        out1_stride_q=out1.stride(1),
        C_=C,
        Q_=Q,
        TOKENS_=TOKENS,
        INV_PATCH_TOKENS_=INV_PATCH_TOKENS,
        BLOCK_Q_=BLOCK_Q,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )

    _finalize_vectors_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial_out0,
        row_value,
        arg106_1,
        out0,
        out2,
        arg106_stride_c=arg106_1.stride(0),
        C_=C,
        N_=N,
        NUM_Q_TILES_=NUM_Q_TILES,
        BLOCK_C_=BLOCK_C,
        BLOCK_TILES_=BLOCK_TILES,
        num_warps=4,
    )
    return out0, out1, out2


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
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
