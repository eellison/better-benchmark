"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ResNeSt backward fragment returned by Repro.forward, fusing the fixed 2x2 avg_pool2d_backward producer, the add/ReLU-gated shared `where`, three sibling channel reductions (`sum(where)`, `sum(where * centered1)`, and `sum(where * centered2)`), both vector outputs, and both dense BN-backward epilogues into one full-scope Triton plan; Inductor currently lowers the structured pool backward, shared masked producer, sibling reductions, and two dependent broadcast epilogues as separable generic schedules with extra materialized traffic and repeated producer work; Inductor cannot do this today because the reduction scheduler does not co-schedule compatible multi-output reductions whose shared producer feeds multiple dependent dense epilogues; the fix is SCHEDULER_FUSION: teach the scheduler to form one multi-accumulator channel-reduction template and sink both BN-backward epilogues into the same fused plan while preserving f32 op order."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


N = 32
C = 256
H = 56
W = 56
HW = H * W
POOL_H = H // 2
POOL_W = W // 2
POOL_HW = POOL_H * POOL_W
NHW = N * HW
TOTAL = N * C * HW
INV_NHW = 9.964923469387754e-06
POOL_SCALE = 0.25
REDUCE_BLOCK = 1024
NUM_REDUCE_BLOCKS = (NHW + REDUCE_BLOCK - 1) // REDUCE_BLOCK
FINAL_BLOCK = 128
EPILOGUE_BLOCK = 256

POOL_SHAPE = (N, C, POOL_H, POOL_W)
POOL_STRIDE = (C * POOL_HW, POOL_HW, POOL_W, 1)
MAIN_SHAPE = (N, C, H, W)
MAIN_STRIDE = (C * HW, HW, W, 1)
MEAN_SHAPE = (1, C, 1, 1)
MEAN_STRIDE = (C, 1, 1, 1)
VEC_SHAPE = (C,)
VEC_STRIDE = (1,)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _add_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _reduce_partials_kernel(
        getitem_36_ptr,
        arg86_ptr,
        getitem_51_ptr,
        full_ptr,
        arg84_ptr,
        arg156_ptr,
        arg82_ptr,
        arg157_ptr,
        partial_where_ptr,
        partial_prod0_ptr,
        partial_prod1_ptr,
        C_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        POOL_W_: tl.constexpr,
        NHW_: tl.constexpr,
        NUM_REDUCE_BLOCKS_: tl.constexpr,
        POOL_SCALE_: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        c = tl.program_id(0)
        block = tl.program_id(1)
        r = block * BLOCK_R + tl.arange(0, BLOCK_R)
        active = r < NHW_
        n = r // HW_
        hw = r - n * HW_
        h = hw // W_
        w = hw - h * W_

        dense_offset = n * (C_ * HW_) + c * HW_ + hw
        pool_hw = (h // 2) * POOL_W_ + (w // 2)
        pool_offset = n * (C_ * ((H_ // 2) * POOL_W_)) + c * ((H_ // 2) * POOL_W_) + pool_hw

        pool_grad = tl.load(getitem_36_ptr + pool_offset, mask=active, other=0.0).to(tl.float32)
        gate = tl.load(arg86_ptr + dense_offset, mask=active, other=0.0).to(tl.float32)
        add_rhs = tl.load(getitem_51_ptr + dense_offset, mask=active, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        x0 = tl.load(arg84_ptr + dense_offset, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(arg82_ptr + dense_offset, mask=active, other=0.0).to(tl.float32)
        mean0 = tl.load(arg156_ptr + c).to(tl.float32)
        mean1 = tl.load(arg157_ptr + c).to(tl.float32)

        pooled = _mul_rn_f32(pool_grad, POOL_SCALE_)
        add_value = _add_rn_f32(pooled, add_rhs)
        where_value = tl.where(gate <= 0.0, full_value, add_value)
        centered0 = _sub_rn_f32(x0, mean0)
        centered1 = _sub_rn_f32(x1, mean1)
        prod0 = _mul_rn_f32(where_value, centered0)
        prod1 = _mul_rn_f32(where_value, centered1)

        partial_offset = c * NUM_REDUCE_BLOCKS_ + block
        tl.store(
            partial_where_ptr + partial_offset,
            tl.sum(tl.where(active, where_value, 0.0), axis=0),
        )
        tl.store(
            partial_prod0_ptr + partial_offset,
            tl.sum(tl.where(active, prod0, 0.0), axis=0),
        )
        tl.store(
            partial_prod1_ptr + partial_offset,
            tl.sum(tl.where(active, prod1, 0.0), axis=0),
        )

    @triton.jit
    def _finalize_scalars_kernel(
        partial_where_ptr,
        partial_prod0_ptr,
        partial_prod1_ptr,
        arg85_ptr,
        arg19_ptr,
        arg83_ptr,
        arg17_ptr,
        coeff_where_ptr,
        coeff_prod0_ptr,
        coeff_weight0_ptr,
        coeff_prod1_ptr,
        coeff_weight1_ptr,
        out_vec0_ptr,
        out_vec1_ptr,
        NUM_REDUCE_BLOCKS_: tl.constexpr,
        BLOCK_B: tl.constexpr,
        INV_NHW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        b = tl.arange(0, BLOCK_B)
        active = b < NUM_REDUCE_BLOCKS_
        offsets = c * NUM_REDUCE_BLOCKS_ + b

        sum_where = tl.sum(
            tl.load(partial_where_ptr + offsets, mask=active, other=0.0).to(tl.float32),
            axis=0,
        )
        sum_prod0 = tl.sum(
            tl.load(partial_prod0_ptr + offsets, mask=active, other=0.0).to(tl.float32),
            axis=0,
        )
        sum_prod1 = tl.sum(
            tl.load(partial_prod1_ptr + offsets, mask=active, other=0.0).to(tl.float32),
            axis=0,
        )

        invstd0 = tl.load(arg85_ptr + c).to(tl.float32)
        weight0 = tl.load(arg19_ptr + c).to(tl.float32)
        invstd1 = tl.load(arg83_ptr + c).to(tl.float32)
        weight1 = tl.load(arg17_ptr + c).to(tl.float32)

        sum_where_scaled = _mul_rn_f32(sum_where, INV_NHW_)

        sum_prod0_scaled = _mul_rn_f32(sum_prod0, INV_NHW_)
        invstd0_sq = _mul_rn_f32(invstd0, invstd0)
        coeff_prod0 = _mul_rn_f32(sum_prod0_scaled, invstd0_sq)
        coeff_weight0 = _mul_rn_f32(invstd0, weight0)
        out_vec0 = _mul_rn_f32(sum_prod0, invstd0)

        sum_prod1_scaled = _mul_rn_f32(sum_prod1, INV_NHW_)
        invstd1_sq = _mul_rn_f32(invstd1, invstd1)
        coeff_prod1 = _mul_rn_f32(sum_prod1_scaled, invstd1_sq)
        coeff_weight1 = _mul_rn_f32(invstd1, weight1)
        out_vec1 = _mul_rn_f32(sum_prod1, invstd1)

        tl.store(coeff_where_ptr + c, sum_where_scaled)
        tl.store(coeff_prod0_ptr + c, coeff_prod0)
        tl.store(coeff_weight0_ptr + c, coeff_weight0)
        tl.store(coeff_prod1_ptr + c, coeff_prod1)
        tl.store(coeff_weight1_ptr + c, coeff_weight1)
        tl.store(out_vec0_ptr + c, out_vec0)
        tl.store(out_vec1_ptr + c, out_vec1)

    @triton.jit
    def _dense_epilogue_kernel(
        getitem_36_ptr,
        arg86_ptr,
        getitem_51_ptr,
        full_ptr,
        arg84_ptr,
        arg156_ptr,
        arg82_ptr,
        arg157_ptr,
        coeff_where_ptr,
        coeff_prod0_ptr,
        coeff_weight0_ptr,
        coeff_prod1_ptr,
        coeff_weight1_ptr,
        out0_ptr,
        out1_ptr,
        C_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        POOL_W_: tl.constexpr,
        TOTAL_: tl.constexpr,
        POOL_SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < TOTAL_
        hw = offsets % HW_
        c = (offsets // HW_) % C_
        n = offsets // (C_ * HW_)
        h = hw // W_
        w = hw - h * W_

        pool_hw = (h // 2) * POOL_W_ + (w // 2)
        pool_offset = n * (C_ * ((H_ // 2) * POOL_W_)) + c * ((H_ // 2) * POOL_W_) + pool_hw

        pool_grad = tl.load(getitem_36_ptr + pool_offset, mask=active, other=0.0).to(tl.float32)
        gate = tl.load(arg86_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        add_rhs = tl.load(getitem_51_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        x0 = tl.load(arg84_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(arg82_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mean0 = tl.load(arg156_ptr + c, mask=active, other=0.0).to(tl.float32)
        mean1 = tl.load(arg157_ptr + c, mask=active, other=0.0).to(tl.float32)
        mean_term = tl.load(coeff_where_ptr + c, mask=active, other=0.0).to(tl.float32)
        coeff0 = tl.load(coeff_prod0_ptr + c, mask=active, other=0.0).to(tl.float32)
        weight0 = tl.load(coeff_weight0_ptr + c, mask=active, other=0.0).to(tl.float32)
        coeff1 = tl.load(coeff_prod1_ptr + c, mask=active, other=0.0).to(tl.float32)
        weight1 = tl.load(coeff_weight1_ptr + c, mask=active, other=0.0).to(tl.float32)

        pooled = _mul_rn_f32(pool_grad, POOL_SCALE_)
        add_value = _add_rn_f32(pooled, add_rhs)
        where_value = tl.where(gate <= 0.0, full_value, add_value)

        centered0 = _sub_rn_f32(x0, mean0)
        correction0 = _mul_rn_f32(centered0, coeff0)
        tmp0 = _sub_rn_f32(where_value, correction0)
        tmp0 = _sub_rn_f32(tmp0, mean_term)
        dense0 = _mul_rn_f32(tmp0, weight0)

        centered1 = _sub_rn_f32(x1, mean1)
        correction1 = _mul_rn_f32(centered1, coeff1)
        tmp1 = _sub_rn_f32(where_value, correction1)
        tmp1 = _sub_rn_f32(tmp1, mean_term)
        dense1 = _mul_rn_f32(tmp1, weight1)

        tl.store(out0_ptr + offsets, dense0, mask=active)
        tl.store(out1_ptr + offsets, dense1, mask=active)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_resnest_dual_bn_backward.py")
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        getitem_36,
        arg86_1,
        getitem_51,
        full,
        arg84_1,
        arg156_1,
        arg85_1,
        arg19_1,
        arg82_1,
        arg157_1,
        arg83_1,
        arg17_1,
    ) = inputs

    getitem_36 = _require_f32_tensor("getitem_36", getitem_36, POOL_SHAPE, POOL_STRIDE)
    arg86_1 = _require_f32_tensor("arg86_1", arg86_1, MAIN_SHAPE, MAIN_STRIDE)
    getitem_51 = _require_f32_tensor("getitem_51", getitem_51, MAIN_SHAPE, MAIN_STRIDE)
    full = _require_f32_tensor("full", full, (), ())
    arg84_1 = _require_f32_tensor("arg84_1", arg84_1, MAIN_SHAPE, MAIN_STRIDE)
    arg156_1 = _require_f32_tensor("arg156_1", arg156_1, MEAN_SHAPE, MEAN_STRIDE)
    arg85_1 = _require_f32_tensor("arg85_1", arg85_1, VEC_SHAPE, VEC_STRIDE)
    arg19_1 = _require_f32_tensor("arg19_1", arg19_1, VEC_SHAPE, VEC_STRIDE)
    arg82_1 = _require_f32_tensor("arg82_1", arg82_1, MAIN_SHAPE, MAIN_STRIDE)
    arg157_1 = _require_f32_tensor("arg157_1", arg157_1, MEAN_SHAPE, MEAN_STRIDE)
    arg83_1 = _require_f32_tensor("arg83_1", arg83_1, VEC_SHAPE, VEC_STRIDE)
    arg17_1 = _require_f32_tensor("arg17_1", arg17_1, VEC_SHAPE, VEC_STRIDE)

    tensors = (
        getitem_36,
        arg86_1,
        getitem_51,
        full,
        arg84_1,
        arg156_1,
        arg85_1,
        arg19_1,
        arg82_1,
        arg157_1,
        arg83_1,
        arg17_1,
    )
    device = getitem_36.device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return tensors


def oracle_forward(inputs):
    """Run the exact Repro.forward scope with shared reductions and epilogues."""
    (
        getitem_36,
        arg86_1,
        getitem_51,
        full,
        arg84_1,
        arg156_1,
        arg85_1,
        arg19_1,
        arg82_1,
        arg157_1,
        arg83_1,
        arg17_1,
    ) = _validate_inputs(inputs)

    device = getitem_36.device
    partial_where = torch.empty_strided(
        (C, NUM_REDUCE_BLOCKS),
        (NUM_REDUCE_BLOCKS, 1),
        device=device,
        dtype=torch.float32,
    )
    partial_prod0 = torch.empty_like(partial_where)
    partial_prod1 = torch.empty_like(partial_where)
    coeff_where = torch.empty_strided(VEC_SHAPE, VEC_STRIDE, device=device, dtype=torch.float32)
    coeff_prod0 = torch.empty_like(coeff_where)
    coeff_weight0 = torch.empty_like(coeff_where)
    coeff_prod1 = torch.empty_like(coeff_where)
    coeff_weight1 = torch.empty_like(coeff_where)
    out_vec0 = torch.empty_like(coeff_where)
    out_vec1 = torch.empty_like(coeff_where)
    out_dense0 = torch.empty_strided(MAIN_SHAPE, MAIN_STRIDE, device=device, dtype=torch.float32)
    out_dense1 = torch.empty_strided(MAIN_SHAPE, MAIN_STRIDE, device=device, dtype=torch.float32)

    _reduce_partials_kernel[(C, NUM_REDUCE_BLOCKS)](
        getitem_36,
        arg86_1,
        getitem_51,
        full,
        arg84_1,
        arg156_1,
        arg82_1,
        arg157_1,
        partial_where,
        partial_prod0,
        partial_prod1,
        C_=C,
        H_=H,
        W_=W,
        HW_=HW,
        POOL_W_=POOL_W,
        NHW_=NHW,
        NUM_REDUCE_BLOCKS_=NUM_REDUCE_BLOCKS,
        POOL_SCALE_=POOL_SCALE,
        BLOCK_R=REDUCE_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _finalize_scalars_kernel[(C,)](
        partial_where,
        partial_prod0,
        partial_prod1,
        arg85_1,
        arg19_1,
        arg83_1,
        arg17_1,
        coeff_where,
        coeff_prod0,
        coeff_weight0,
        coeff_prod1,
        coeff_weight1,
        out_vec0,
        out_vec1,
        NUM_REDUCE_BLOCKS_=NUM_REDUCE_BLOCKS,
        BLOCK_B=FINAL_BLOCK,
        INV_NHW_=INV_NHW,
        num_warps=4,
        num_stages=4,
    )
    _dense_epilogue_kernel[(triton.cdiv(TOTAL, EPILOGUE_BLOCK),)](
        getitem_36,
        arg86_1,
        getitem_51,
        full,
        arg84_1,
        arg156_1,
        arg82_1,
        arg157_1,
        coeff_where,
        coeff_prod0,
        coeff_weight0,
        coeff_prod1,
        coeff_weight1,
        out_dense0,
        out_dense1,
        C_=C,
        H_=H,
        W_=W,
        HW_=HW,
        POOL_W_=POOL_W,
        TOTAL_=TOTAL,
        POOL_SCALE_=POOL_SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    return out_dense0, out_vec0, out_dense1, out_vec1


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
