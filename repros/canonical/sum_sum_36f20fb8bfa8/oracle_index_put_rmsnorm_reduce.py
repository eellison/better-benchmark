"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle performs the masked MoE index_put(accumulate=True) scatter-reduce into the 2048x2048 matrix and feeds that accumulated matrix through both the column reduction and RMSNorm-backward transpose outputs, whereas Inductor currently materializes the generic scatter result and schedules the sibling reductions and transpose epilogue as separate kernels around it; Inductor cannot do this today because scheduler/codegen cannot represent an index_put scatter-reduce producer that is reused by dependent row reductions, column reductions, and a strided transpose store in one planned lowering; the fix is SCATTER_REDUCE: add a scatter-reduce lowering that keeps the accumulated rows available to the downstream RMSNorm reductions and transpose epilogue during scheduling."""
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

REPRO_ID = "sum_sum_36f20fb8bfa8"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_Qwen_Qwen3-30B-A3B_001_3b4ca287"

N_ROWS = 2048
HIDDEN = 2048
N_SCATTER = 16384
BLOCK_H = 128
BLOCK_M = 128
NUM_H_BLOCKS = HIDDEN // BLOCK_H
NUM_M_BLOCKS = N_ROWS // BLOCK_M


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
    def _partial_reduce_kernel(
        scatter_ptr,
        mm_ptr,
        activation_ptr,
        inv_ptr,
        partial_col_ptr,
        BLOCK_M_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
        HIDDEN_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        h_offsets = pid_h * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        offsets = m_offsets[:, None] * HIDDEN_ + h_offsets[None, :]

        scatter = tl.load(scatter_ptr + offsets).to(tl.float32)
        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        x = (scatter + mm).to(tl.bfloat16).to(tl.float32)

        activation = tl.load(activation_ptr + offsets).to(tl.float32)
        inv = tl.load(inv_ptr + m_offsets).to(tl.float32)

        normalized = (activation * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        col_terms = (x * normalized).to(tl.bfloat16).to(tl.float32)
        col_sums = tl.sum(col_terms, axis=0)
        tl.store(partial_col_ptr + pid_m * HIDDEN_ + h_offsets, col_sums)

    @triton.jit
    def _row_sum_kernel(
        scatter_ptr,
        mm_ptr,
        weight_ptr,
        activation_ptr,
        row_sum_ptr,
        BLOCK_H_FULL_: tl.constexpr,
        HIDDEN_: tl.constexpr,
    ):
        m = tl.program_id(0)
        h_offsets = tl.arange(0, BLOCK_H_FULL_)
        offsets = m * HIDDEN_ + h_offsets

        scatter = tl.load(scatter_ptr + offsets).to(tl.float32)
        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        x = (scatter + mm).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(weight_ptr + h_offsets).to(tl.float32)
        activation = tl.load(activation_ptr + offsets).to(tl.float32)

        weighted = (x * weight).to(tl.bfloat16).to(tl.float32)
        row_sum = tl.sum(weighted * activation, axis=0)
        tl.store(row_sum_ptr + m, row_sum)

    @triton.jit
    def _finalize_col_kernel(
        partial_col_ptr,
        out0_ptr,
        BLOCK_H_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        NUM_M_BLOCKS_: tl.constexpr,
    ):
        h_offsets = tl.program_id(0) * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        m_blocks = tl.arange(0, NUM_M_BLOCKS_)
        values = tl.load(partial_col_ptr + m_blocks[:, None] * HIDDEN_ + h_offsets[None, :]).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(out0_ptr + h_offsets, sums)

    @triton.jit
    def _rmsnorm_transpose_kernel(
        scatter_ptr,
        mm_ptr,
        weight_ptr,
        activation_ptr,
        inv_ptr,
        residual_ptr,
        row_sum_ptr,
        out_base_ptr,
        BLOCK_M_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
        HIDDEN_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        h_offsets = pid_h * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        offsets = m_offsets[:, None] * HIDDEN_ + h_offsets[None, :]

        scatter = tl.load(scatter_ptr + offsets).to(tl.float32)
        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        x = (scatter + mm).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(weight_ptr + h_offsets).to(tl.float32)
        activation = tl.load(activation_ptr + offsets).to(tl.float32)
        inv = tl.load(inv_ptr + m_offsets).to(tl.float32)
        row_sum = tl.load(row_sum_ptr + m_offsets).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)

        weighted = (x * weight[None, :]).to(tl.bfloat16).to(tl.float32)
        mul_tensor_4 = weighted * inv[:, None]
        pow_tensor_scalar = inv * inv * inv
        mul_scalar = row_sum * -0.5
        mul_tensor_5 = mul_scalar * pow_tensor_scalar
        div_scalar = mul_tensor_5 / HIDDEN_
        mul_scalar_1 = activation * 2.0
        mul_tensor_6 = div_scalar[:, None] * mul_scalar_1
        update = mul_tensor_4 + mul_tensor_6
        update_bf16 = update.to(tl.bfloat16).to(tl.float32)
        out = (residual + update_bf16).to(tl.bfloat16)
        tl.store(out_base_ptr + offsets, out)


def oracle_index_put_rmsnorm_reduce(
    arg171_1: torch.Tensor,
    full_3: torch.Tensor,
    _grouped_mm_3: torch.Tensor,
    arg169_1: torch.Tensor,
    mm_3: torch.Tensor,
    arg42_1: torch.Tensor,
    arg159_1: torch.Tensor,
    arg160_1: torch.Tensor,
    convert_element_type_5: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm_3.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    where_self = torch.ops.aten.where.self(arg171_1, full_3, _grouped_mm_3)
    scatter = torch.ops.aten.index_put.default(
        torch.zeros((N_ROWS, HIDDEN), device=mm_3.device, dtype=mm_3.dtype),
        [arg169_1],
        where_self,
        True,
    )
    out0 = torch.empty((HIDDEN,), device=mm_3.device, dtype=mm_3.dtype)
    out_base = torch.empty((N_ROWS, HIDDEN), device=mm_3.device, dtype=mm_3.dtype)
    partial_col = torch.empty((NUM_M_BLOCKS, HIDDEN), device=mm_3.device, dtype=torch.float32)
    row_sum = torch.empty((N_ROWS,), device=mm_3.device, dtype=torch.float32)

    _partial_reduce_kernel[(NUM_H_BLOCKS, NUM_M_BLOCKS)](
        scatter,
        mm_3,
        arg159_1,
        arg160_1,
        partial_col,
        BLOCK_M_=BLOCK_M,
        BLOCK_H_=BLOCK_H,
        HIDDEN_=HIDDEN,
        num_warps=4,
    )
    _finalize_col_kernel[(NUM_H_BLOCKS,)](
        partial_col,
        out0,
        BLOCK_H_=BLOCK_H,
        HIDDEN_=HIDDEN,
        NUM_M_BLOCKS_=NUM_M_BLOCKS,
        num_warps=1,
    )
    _row_sum_kernel[(N_ROWS,)](
        scatter,
        mm_3,
        arg42_1,
        arg159_1,
        row_sum,
        BLOCK_H_FULL_=HIDDEN,
        HIDDEN_=HIDDEN,
        num_warps=8,
    )
    _rmsnorm_transpose_kernel[(NUM_H_BLOCKS, NUM_M_BLOCKS)](
        scatter,
        mm_3,
        arg42_1,
        arg159_1,
        arg160_1,
        convert_element_type_5,
        row_sum,
        out_base,
        BLOCK_M_=BLOCK_M,
        BLOCK_H_=BLOCK_H,
        HIDDEN_=HIDDEN,
        num_warps=4,
    )
    return out0, out_base.t()


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


@oracle_impl(hardware="H100", shapes="(T([16384, 1], b8), T([], bf16), T([16384, 2048], bf16), T([16384], i64, gen=Index(2048)), T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([4, 512, 2048], bf16), S([4, 512, 2048]), S([2048]), S([4, 512, 2048]), S([2048, 2048]))")
def oracle_forward(inputs):
    return oracle_index_put_rmsnorm_reduce(*inputs)


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
