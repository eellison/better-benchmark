"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ResNeSt BN-backward-style captured scope, including the expanded grouped producer, ReLU mask select, both channel reductions, the scale-gradient vector, and the full dense epilogue, while keeping the two sibling channel reductions in one Triton pass with `sum(where)` and `sum(where * centered)` accumulators and recomputing the cheap producer for the epilogue instead of materializing it; Inductor does not co-schedule this multi-output reduction today because the scheduler fails to fuse sibling reductions over the same `numel/rnumel` and shared viewed/expanded masked producer once their summaries feed different epilogues; the fix is SCHEDULER_FUSION: teach the reduction scheduler to recognize shared-input sibling channel reductions with equal reduction domains and emit one multi-accumulator reduction plus the dependent dense epilogue."""
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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

N = 32
GROUPS = 2
GROUP_C = 64
C = GROUPS * GROUP_C
H = 56
W = 56
HW = H * W
TOTAL = N * C * HW
REDUCE_N = N * HW
REDUCE_SCALE = 9.964923469387754e-06
POOL_DENOM = 3136.0
REDUCE_GROUPS = 4
GROUP_N = N // REDUCE_GROUPS
GROUP_REDUCE = GROUP_N * HW
REDUCE_BLOCK = 1024
POINTWISE_BLOCK = 256

DIV_SHAPE = (N, GROUPS, 1, GROUP_C)
DIV_STRIDE = (GROUPS * GROUP_C, GROUP_C, GROUP_C, 1)
EXPAND_SHAPE = (N, GROUPS, GROUP_C, H, W)
EXPAND_STRIDE = (GROUP_C * HW, 0, HW, W, 1)
HALF_CHANNEL_SHAPE = (N, GROUP_C, 1, 1)
HALF_CHANNEL_STRIDE = (GROUP_C, 1, 1, 1)
ACT_SHAPE = (N, C, H, W)
ACT_STRIDE = (C * HW, HW, W, 1)
CHANNEL_4D_SHAPE = (1, C, 1, 1)
CHANNEL_4D_STRIDE = (C, 1, 1, 1)
CHANNEL_SHAPE = (C,)
CHANNEL_STRIDE = (1,)

SHAPE_PARAMS = (
    [32, -1],
    [32, -1, 1, 1],
    [32, 2, 64, 1, 1],
    [32, 64, 56, 56],
    [32, 2, 64, 56, 56],
    [32, 128, 56, 56],
)


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
    def _div_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _partial_dual_reduce_kernel(
        div_ptr,
        expand_ptr,
        getitem_ptr,
        relu_ptr,
        full_ptr,
        mean_ptr,
        arg73_ptr,
        partial_where_ptr,
        partial_prod_ptr,
        GROUP_C_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        GROUP_N_: tl.constexpr,
        GROUP_REDUCE_: tl.constexpr,
        REDUCE_GROUPS_: tl.constexpr,
        POOL_DENOM_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        reduce_group = tl.program_id(1)
        k_offsets = tl.arange(0, BLOCK_K)
        acc_where = tl.full((BLOCK_K,), 0.0, tl.float32)
        acc_prod = tl.full((BLOCK_K,), 0.0, tl.float32)

        group = channel // GROUP_C_
        inner_c = channel - group * GROUP_C_
        full_value = tl.load(full_ptr).to(tl.float32)
        mean = tl.load(mean_ptr + channel).to(tl.float32)

        for base in tl.range(0, GROUP_REDUCE_, BLOCK_K):
            k = base + k_offsets
            valid = k < GROUP_REDUCE_
            n_in_group = k // HW_
            n = reduce_group * GROUP_N_ + n_in_group
            spatial = k - n_in_group * HW_
            dense_offset = n * (C_ * HW_) + channel * HW_ + spatial
            half_offset = n * (GROUP_C_ * HW_) + inner_c * HW_ + spatial
            div_offset = n * (2 * GROUP_C_) + group * GROUP_C_ + inner_c
            getitem_offset = n * GROUP_C_ + inner_c

            expand_value = tl.load(expand_ptr + half_offset, mask=valid, other=0.0).to(tl.float32)
            div_value = tl.load(div_ptr + div_offset, mask=valid, other=0.0).to(tl.float32)
            pooled = tl.load(getitem_ptr + getitem_offset, mask=valid, other=0.0).to(tl.float32)
            relu_value = tl.load(relu_ptr + dense_offset, mask=valid, other=1.0).to(tl.float32)
            x_value = tl.load(arg73_ptr + dense_offset, mask=valid, other=0.0).to(tl.float32)

            pooled = _div_rn_f32(pooled, POOL_DENOM_)
            product = _mul_rn_f32(expand_value, div_value)
            add_value = _add_rn_f32(product, pooled)
            where_value = tl.where(relu_value <= 0.0, full_value, add_value)
            centered = _sub_rn_f32(x_value, mean)
            prod = _mul_rn_f32(where_value, centered)

            acc_where = _add_rn_f32(acc_where, tl.where(valid, where_value, 0.0))
            acc_prod = _add_rn_f32(acc_prod, tl.where(valid, prod, 0.0))

        partial_offset = channel * REDUCE_GROUPS_ + reduce_group
        tl.store(partial_where_ptr + partial_offset, tl.sum(acc_where, axis=0))
        tl.store(partial_prod_ptr + partial_offset, tl.sum(acc_prod, axis=0))

    @triton.jit
    def _finalize_kernel(
        partial_where_ptr,
        partial_prod_ptr,
        invstd_ptr,
        sum_where_ptr,
        sum_prod_ptr,
        out_vec_ptr,
        REDUCE_GROUPS_: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        channel = tl.program_id(0)
        groups = tl.arange(0, BLOCK_G)
        mask = groups < REDUCE_GROUPS_
        partial_offsets = channel * REDUCE_GROUPS_ + groups

        where_values = tl.load(partial_where_ptr + partial_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        prod_values = tl.load(partial_prod_ptr + partial_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        sum_where = tl.sum(where_values, axis=0).to(tl.float32)
        sum_prod = tl.sum(prod_values, axis=0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        out_vec = _mul_rn_f32(sum_prod, invstd)

        tl.store(sum_where_ptr + channel, sum_where)
        tl.store(sum_prod_ptr + channel, sum_prod)
        tl.store(out_vec_ptr + channel, out_vec)

    @triton.jit
    def _dense_epilogue_kernel(
        div_ptr,
        expand_ptr,
        getitem_ptr,
        relu_ptr,
        full_ptr,
        mean_ptr,
        arg73_ptr,
        invstd_ptr,
        weight_ptr,
        sum_where_ptr,
        sum_prod_ptr,
        out_ptr,
        GROUP_C_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
        POOL_DENOM_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL_
        channel = (offsets // HW_) % C_
        n = offsets // (C_ * HW_)
        spatial = offsets % HW_
        group = channel // GROUP_C_
        inner_c = channel - group * GROUP_C_
        half_offset = n * (GROUP_C_ * HW_) + inner_c * HW_ + spatial
        div_offset = n * (2 * GROUP_C_) + group * GROUP_C_ + inner_c
        getitem_offset = n * GROUP_C_ + inner_c

        expand_value = tl.load(expand_ptr + half_offset, mask=mask, other=0.0).to(tl.float32)
        div_value = tl.load(div_ptr + div_offset, mask=mask, other=0.0).to(tl.float32)
        pooled = tl.load(getitem_ptr + getitem_offset, mask=mask, other=0.0).to(tl.float32)
        relu_value = tl.load(relu_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        x_value = tl.load(arg73_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        sum_where = tl.load(sum_where_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        sum_prod = tl.load(sum_prod_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        pooled = _div_rn_f32(pooled, POOL_DENOM_)
        product = _mul_rn_f32(expand_value, div_value)
        add_value = _add_rn_f32(product, pooled)
        where_value = tl.where(relu_value <= 0.0, tl.load(full_ptr).to(tl.float32), add_value)
        centered = _sub_rn_f32(x_value, mean)
        prod_mean = _mul_rn_f32(sum_prod, REDUCE_SCALE_)
        invstd_sq = _mul_rn_f32(invstd, invstd)
        coeff = _mul_rn_f32(prod_mean, invstd_sq)
        correction = _mul_rn_f32(centered, coeff)
        out = _sub_rn_f32(where_value, correction)
        where_mean = _mul_rn_f32(sum_where, REDUCE_SCALE_)
        out = _sub_rn_f32(out, where_mean)
        out_scale = _mul_rn_f32(invstd, weight)
        out = _mul_rn_f32(out, out_scale)
        tl.store(out_ptr + offsets, out, mask=mask)


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
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 15:
        raise ValueError(f"{REPRO_ID} expects 15 inputs, got {len(inputs)}")

    (
        div_7,
        expand_10,
        getitem_63,
        relu_4,
        full,
        arg74_1,
        arg73_1,
        arg75_1,
        arg11_1,
        *shape_params,
    ) = inputs
    if shape_params != list(SHAPE_PARAMS):
        raise ValueError(f"unexpected shape parameters: {shape_params!r}")

    div_7 = _require_f32_tensor("div_7", div_7, DIV_SHAPE, DIV_STRIDE)
    expand_10 = _require_f32_tensor("expand_10", expand_10, EXPAND_SHAPE, EXPAND_STRIDE)
    getitem_63 = _require_f32_tensor("getitem_63", getitem_63, HALF_CHANNEL_SHAPE, HALF_CHANNEL_STRIDE)
    relu_4 = _require_f32_tensor("relu_4", relu_4, ACT_SHAPE, ACT_STRIDE)
    full = _require_f32_tensor("full", full, (), ())
    arg74_1 = _require_f32_tensor("arg74_1", arg74_1, CHANNEL_4D_SHAPE, CHANNEL_4D_STRIDE)
    arg73_1 = _require_f32_tensor("arg73_1", arg73_1, ACT_SHAPE, ACT_STRIDE)
    arg75_1 = _require_f32_tensor("arg75_1", arg75_1, CHANNEL_4D_SHAPE, CHANNEL_4D_STRIDE)
    arg11_1 = _require_f32_tensor("arg11_1", arg11_1, CHANNEL_SHAPE, CHANNEL_STRIDE)

    tensors = (div_7, expand_10, getitem_63, relu_4, full, arg74_1, arg73_1, arg75_1, arg11_1)
    device = div_7.device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return div_7, expand_10, getitem_63, relu_4, full, arg74_1, arg73_1, arg75_1, arg11_1


@oracle_impl(hardware="H100", shapes="(T([32, 2, 1, 64], f32), T([32, 2, 64, 56, 56], f32, stride=(200704, 0, 3136, 56, 1)), T([32, 64, 1, 1], f32), T([32, 128, 56, 56], f32), T([], f32), T([1, 128, 1, 1], f32), T([32, 128, 56, 56], f32), T([1, 128, 1, 1], f32), T([128], f32), S([32, -1]), S([32, -1, 1, 1]), S([32, 2, 64, 1, 1]), S([32, 64, 56, 56]), S([32, 2, 64, 56, 56]), S([32, 128, 56, 56]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    (
        div_7,
        expand_10,
        getitem_63,
        relu_4,
        full,
        arg74_1,
        arg73_1,
        arg75_1,
        arg11_1,
    ) = _validate_inputs(inputs)

    device = div_7.device
    partial_where = torch.empty_strided(
        (C, REDUCE_GROUPS),
        (REDUCE_GROUPS, 1),
        device=device,
        dtype=torch.float32,
    )
    partial_prod = torch.empty_like(partial_where)
    sum_where = torch.empty_strided(CHANNEL_SHAPE, CHANNEL_STRIDE, device=device, dtype=torch.float32)
    sum_prod = torch.empty_like(sum_where)
    out_tensor = torch.empty_strided(ACT_SHAPE, ACT_STRIDE, device=device, dtype=torch.float32)
    out_vec = torch.empty_strided(CHANNEL_SHAPE, CHANNEL_STRIDE, device=device, dtype=torch.float32)

    _partial_dual_reduce_kernel[(C, REDUCE_GROUPS)](
        div_7,
        expand_10,
        getitem_63,
        relu_4,
        full,
        arg74_1,
        arg73_1,
        partial_where,
        partial_prod,
        GROUP_C_=GROUP_C,
        C_=C,
        HW_=HW,
        GROUP_N_=GROUP_N,
        GROUP_REDUCE_=GROUP_REDUCE,
        REDUCE_GROUPS_=REDUCE_GROUPS,
        POOL_DENOM_=POOL_DENOM,
        BLOCK_K=REDUCE_BLOCK,
        num_warps=8,
        num_stages=1,
    )
    _finalize_kernel[(C,)](
        partial_where,
        partial_prod,
        arg75_1,
        sum_where,
        sum_prod,
        out_vec,
        REDUCE_GROUPS_=REDUCE_GROUPS,
        BLOCK_G=REDUCE_GROUPS,
        num_warps=1,
        num_stages=1,
    )
    _dense_epilogue_kernel[(triton.cdiv(TOTAL, POINTWISE_BLOCK),)](
        div_7,
        expand_10,
        getitem_63,
        relu_4,
        full,
        arg74_1,
        arg73_1,
        arg75_1,
        arg11_1,
        sum_where,
        sum_prod,
        out_tensor,
        GROUP_C_=GROUP_C,
        C_=C,
        HW_=HW,
        TOTAL_=TOTAL,
        REDUCE_SCALE_=REDUCE_SCALE,
        POOL_DENOM_=POOL_DENOM,
        BLOCK_N=POINTWISE_BLOCK,
        num_warps=4,
        num_stages=1,
    )
    return out_tensor, out_vec


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
