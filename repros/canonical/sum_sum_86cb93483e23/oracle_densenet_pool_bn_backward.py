"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet backward tail returned by Repro.forward, sharing the sibling `sum(where)` and `sum(where * centered)` channel reductions, preserving the sequential 12-input sliced accumulation, and writing the dependent batch-norm epilogue directly into the fixed 2x2 avg_pool2d_backward output; Inductor currently handles the sliced residual producer, paired reductions, finalized scalar epilogue, and pool-backward expansion as separable generic regions with extra intermediate traffic; Inductor cannot do this today because the scheduler does not form one full-scope multi-output reduction plan whose finalized channel scalars feed a layout-changing pool-backward consumer; the fix is SCHEDULER_FUSION: teach reduction scheduling to keep compatible sibling channel reductions, the sliced residual producer, and the structured 2x2 pool-backward epilogue in one fused template."""
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


N = 64
C = 128
H = 28
W = 28
HW = H * W
NHW = N * HW
PARENT_NUMEL = N * C * HW
OUT_H = 56
OUT_W = 56
OUT_HW = OUT_H * OUT_W
INV_NHW = 1.992984693877551e-05
REDUCE_BLOCK = 1024
NUM_REDUCE_BLOCKS = (NHW + REDUCE_BLOCK - 1) // REDUCE_BLOCK
FINAL_BLOCK = 64
EPILOGUE_BLOCK = 256

IN_CHANNELS = (512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160)
MAIN_SHAPE = (N, C, H, W)
MAIN_STRIDE = (C * HW, HW, W, 1)
MEAN_SHAPE = (1, C, 1, 1)
MEAN_STRIDE = (C, 1, 1, 1)
VEC_SHAPE = (C,)
VEC_STRIDE = (1,)
OUT1_SHAPE = (N, C, OUT_H, OUT_W)
OUT1_STRIDE = (C * OUT_HW, OUT_HW, OUT_W, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

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
    def _reduce_where_partials_kernel(
        arg290_ptr,
        full_ptr,
        getitem_315_ptr,
        arg288_ptr,
        arg717_ptr,
        partial_where_ptr,
        partial_mul_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        NUM_REDUCE_BLOCKS_: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        c = tl.program_id(0)
        block = tl.program_id(1)
        r = block * BLOCK_R + tl.arange(0, BLOCK_R)
        active = r < NHW_
        n = r // HW_
        hw = r - n * HW_
        offset = n * (C_ * HW_) + c * HW_ + hw

        mask_value = tl.load(arg290_ptr + offset, mask=active, other=0.0).to(tl.float32)
        where_rhs = tl.load(getitem_315_ptr + offset, mask=active, other=0.0).to(tl.float32)
        x_value = tl.load(arg288_ptr + offset, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(arg717_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        where_value = tl.where(mask_value <= 0.0, full_value, where_rhs)
        centered = _sub_rn_f32(x_value, mean)
        product = _mul_rn_f32(where_value, centered)

        partial_offset = c * NUM_REDUCE_BLOCKS_ + block
        tl.store(partial_where_ptr + partial_offset, tl.sum(tl.where(active, where_value, 0.0), axis=0))
        tl.store(partial_mul_ptr + partial_offset, tl.sum(tl.where(active, product, 0.0), axis=0))

    @triton.jit
    def _finalize_scalars_kernel(
        partial_where_ptr,
        partial_mul_ptr,
        arg289_ptr,
        arg30_ptr,
        coeff_where_ptr,
        coeff_mul_ptr,
        coeff_weight_ptr,
        out0_ptr,
        NUM_REDUCE_BLOCKS_: tl.constexpr,
        BLOCK_B: tl.constexpr,
        INV_NHW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        b = tl.arange(0, BLOCK_B)
        mask = b < NUM_REDUCE_BLOCKS_
        offsets = c * NUM_REDUCE_BLOCKS_ + b

        sum_where = tl.sum(tl.load(partial_where_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum_mul = tl.sum(tl.load(partial_mul_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        invstd = tl.load(arg289_ptr + c).to(tl.float32)
        affine_weight = tl.load(arg30_ptr + c).to(tl.float32)

        sum_where_scaled = _mul_rn_f32(sum_where, INV_NHW_)
        sum_mul_scaled = _mul_rn_f32(sum_mul, INV_NHW_)
        invstd_sq = _mul_rn_f32(invstd, invstd)
        mul_coeff = _mul_rn_f32(sum_mul_scaled, invstd_sq)
        out_weight = _mul_rn_f32(invstd, affine_weight)
        out0 = _mul_rn_f32(sum_mul, invstd)

        tl.store(coeff_where_ptr + c, sum_where_scaled)
        tl.store(coeff_mul_ptr + c, mul_coeff)
        tl.store(coeff_weight_ptr + c, out_weight)
        tl.store(out0_ptr + c, out0)

    @triton.jit
    def _pool_epilogue_kernel(
        out1_ptr,
        mul_747_ptr,
        mul_765_ptr,
        mul_783_ptr,
        mul_801_ptr,
        mul_819_ptr,
        mul_837_ptr,
        mul_855_ptr,
        mul_873_ptr,
        mul_891_ptr,
        mul_909_ptr,
        mul_927_ptr,
        mul_945_ptr,
        arg290_ptr,
        full_ptr,
        getitem_315_ptr,
        arg288_ptr,
        arg717_ptr,
        coeff_where_ptr,
        coeff_mul_ptr,
        coeff_weight_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        OUT_W_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        PARENT_NUMEL_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < PARENT_NUMEL_
        hw = offsets % HW_
        c = (offsets // HW_) % C_
        n = offsets // (C_ * HW_)

        local = c * HW_ + hw
        x0 = tl.load(mul_747_ptr + n * (512 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(mul_765_ptr + n * (480 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x2 = tl.load(mul_783_ptr + n * (448 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x3 = tl.load(mul_801_ptr + n * (416 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x4 = tl.load(mul_819_ptr + n * (384 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x5 = tl.load(mul_837_ptr + n * (352 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x6 = tl.load(mul_855_ptr + n * (320 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x7 = tl.load(mul_873_ptr + n * (288 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x8 = tl.load(mul_891_ptr + n * (256 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x9 = tl.load(mul_909_ptr + n * (224 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x10 = tl.load(mul_927_ptr + n * (192 * HW_) + local, mask=active, other=0.0).to(tl.float32)
        x11 = tl.load(mul_945_ptr + n * (160 * HW_) + local, mask=active, other=0.0).to(tl.float32)

        acc = _add_rn_f32(x0, x1)
        acc = _add_rn_f32(acc, x2)
        acc = _add_rn_f32(acc, x3)
        acc = _add_rn_f32(acc, x4)
        acc = _add_rn_f32(acc, x5)
        acc = _add_rn_f32(acc, x6)
        acc = _add_rn_f32(acc, x7)
        acc = _add_rn_f32(acc, x8)
        acc = _add_rn_f32(acc, x9)
        acc = _add_rn_f32(acc, x10)
        acc = _add_rn_f32(acc, x11)

        mask_value = tl.load(arg290_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        where_rhs = tl.load(getitem_315_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        x_value = tl.load(arg288_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(arg717_ptr + c, mask=active, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        mean_term = tl.load(coeff_where_ptr + c, mask=active, other=0.0).to(tl.float32)
        mul_coeff = tl.load(coeff_mul_ptr + c, mask=active, other=0.0).to(tl.float32)
        out_weight = tl.load(coeff_weight_ptr + c, mask=active, other=0.0).to(tl.float32)

        where_value = tl.where(mask_value <= 0.0, full_value, where_rhs)
        centered = _sub_rn_f32(x_value, mean)
        correction = _mul_rn_f32(centered, mul_coeff)
        tmp = _sub_rn_f32(where_value, correction)
        tmp = _sub_rn_f32(tmp, mean_term)
        bn_epilogue = _mul_rn_f32(tmp, out_weight)
        pooled_parent = _add_rn_f32(acc, bn_epilogue) / 4.0

        h = hw // (OUT_W_ // 2)
        w = hw - h * (OUT_W_ // 2)
        out_base = n * (C_ * OUT_HW_) + c * OUT_HW_ + h * (2 * OUT_W_) + w * 2
        tl.store(out1_ptr + out_base, pooled_parent, mask=active)
        tl.store(out1_ptr + out_base + 1, pooled_parent, mask=active)
        tl.store(out1_ptr + out_base + OUT_W_, pooled_parent, mask=active)
        tl.store(out1_ptr + out_base + OUT_W_ + 1, pooled_parent, mask=active)


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
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_densenet_pool_bn_backward.py")
    if len(inputs) != 20:
        raise ValueError(f"{REPRO_ID} expects 20 inputs, got {len(inputs)}")

    (
        mul_747,
        mul_765,
        mul_783,
        mul_801,
        mul_819,
        mul_837,
        mul_855,
        mul_873,
        mul_891,
        mul_909,
        mul_927,
        mul_945,
        arg290_1,
        full,
        getitem_315,
        arg288_1,
        arg717_1,
        arg289_1,
        arg30_1,
        arg287_1,
    ) = inputs

    slice_inputs = (
        _require_f32_tensor("mul_747", mul_747, (N, IN_CHANNELS[0], H, W), (IN_CHANNELS[0] * HW, HW, W, 1)),
        _require_f32_tensor("mul_765", mul_765, (N, IN_CHANNELS[1], H, W), (IN_CHANNELS[1] * HW, HW, W, 1)),
        _require_f32_tensor("mul_783", mul_783, (N, IN_CHANNELS[2], H, W), (IN_CHANNELS[2] * HW, HW, W, 1)),
        _require_f32_tensor("mul_801", mul_801, (N, IN_CHANNELS[3], H, W), (IN_CHANNELS[3] * HW, HW, W, 1)),
        _require_f32_tensor("mul_819", mul_819, (N, IN_CHANNELS[4], H, W), (IN_CHANNELS[4] * HW, HW, W, 1)),
        _require_f32_tensor("mul_837", mul_837, (N, IN_CHANNELS[5], H, W), (IN_CHANNELS[5] * HW, HW, W, 1)),
        _require_f32_tensor("mul_855", mul_855, (N, IN_CHANNELS[6], H, W), (IN_CHANNELS[6] * HW, HW, W, 1)),
        _require_f32_tensor("mul_873", mul_873, (N, IN_CHANNELS[7], H, W), (IN_CHANNELS[7] * HW, HW, W, 1)),
        _require_f32_tensor("mul_891", mul_891, (N, IN_CHANNELS[8], H, W), (IN_CHANNELS[8] * HW, HW, W, 1)),
        _require_f32_tensor("mul_909", mul_909, (N, IN_CHANNELS[9], H, W), (IN_CHANNELS[9] * HW, HW, W, 1)),
        _require_f32_tensor("mul_927", mul_927, (N, IN_CHANNELS[10], H, W), (IN_CHANNELS[10] * HW, HW, W, 1)),
        _require_f32_tensor("mul_945", mul_945, (N, IN_CHANNELS[11], H, W), (IN_CHANNELS[11] * HW, HW, W, 1)),
    )
    arg290_1 = _require_f32_tensor("arg290_1", arg290_1, MAIN_SHAPE, MAIN_STRIDE)
    full = _require_f32_tensor("full", full, (), ())
    getitem_315 = _require_f32_tensor("getitem_315", getitem_315, MAIN_SHAPE, MAIN_STRIDE)
    arg288_1 = _require_f32_tensor("arg288_1", arg288_1, MAIN_SHAPE, MAIN_STRIDE)
    arg717_1 = _require_f32_tensor("arg717_1", arg717_1, MEAN_SHAPE, MEAN_STRIDE)
    arg289_1 = _require_f32_tensor("arg289_1", arg289_1, VEC_SHAPE, VEC_STRIDE)
    arg30_1 = _require_f32_tensor("arg30_1", arg30_1, VEC_SHAPE, VEC_STRIDE)
    arg287_1 = _require_f32_tensor("arg287_1", arg287_1, OUT1_SHAPE, OUT1_STRIDE)

    device = slice_inputs[0].device
    all_tensors = (
        *slice_inputs,
        arg290_1,
        full,
        getitem_315,
        arg288_1,
        arg717_1,
        arg289_1,
        arg30_1,
        arg287_1,
    )
    if any(t.device != device for t in all_tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        *slice_inputs,
        arg290_1,
        full,
        getitem_315,
        arg288_1,
        arg717_1,
        arg289_1,
        arg30_1,
        arg287_1,
    )


@oracle_impl(hardware="H100", shapes="(T([64, 512, 28, 28], f32), T([64, 480, 28, 28], f32), T([64, 448, 28, 28], f32), T([64, 416, 28, 28], f32), T([64, 384, 28, 28], f32), T([64, 352, 28, 28], f32), T([64, 320, 28, 28], f32), T([64, 288, 28, 28], f32), T([64, 256, 28, 28], f32), T([64, 224, 28, 28], f32), T([64, 192, 28, 28], f32), T([64, 160, 28, 28], f32), T([64, 128, 28, 28], f32), T([], f32), T([64, 128, 28, 28], f32), T([64, 128, 28, 28], f32), T([1, 128, 1, 1], f32), T([128], f32), T([128], f32), T([64, 128, 56, 56], f32))")
def oracle_forward(inputs):
    """Run the exact Repro.forward scope with shared reductions and pool expansion."""
    (
        mul_747,
        mul_765,
        mul_783,
        mul_801,
        mul_819,
        mul_837,
        mul_855,
        mul_873,
        mul_891,
        mul_909,
        mul_927,
        mul_945,
        arg290_1,
        full,
        getitem_315,
        arg288_1,
        arg717_1,
        arg289_1,
        arg30_1,
        _arg287_1,
    ) = _validate_inputs(inputs)

    device = mul_747.device
    partial_where = torch.empty_strided(
        (C, NUM_REDUCE_BLOCKS),
        (NUM_REDUCE_BLOCKS, 1),
        device=device,
        dtype=torch.float32,
    )
    partial_mul = torch.empty_strided(
        (C, NUM_REDUCE_BLOCKS),
        (NUM_REDUCE_BLOCKS, 1),
        device=device,
        dtype=torch.float32,
    )
    coeff_where = torch.empty_strided(VEC_SHAPE, VEC_STRIDE, device=device, dtype=torch.float32)
    coeff_mul = torch.empty_strided(VEC_SHAPE, VEC_STRIDE, device=device, dtype=torch.float32)
    coeff_weight = torch.empty_strided(VEC_SHAPE, VEC_STRIDE, device=device, dtype=torch.float32)
    out0 = torch.empty_strided(VEC_SHAPE, VEC_STRIDE, device=device, dtype=torch.float32)
    out1 = torch.empty_strided(OUT1_SHAPE, OUT1_STRIDE, device=device, dtype=torch.float32)

    _reduce_where_partials_kernel[(C, NUM_REDUCE_BLOCKS)](
        arg290_1,
        full,
        getitem_315,
        arg288_1,
        arg717_1,
        partial_where,
        partial_mul,
        C_=C,
        HW_=HW,
        NHW_=NHW,
        NUM_REDUCE_BLOCKS_=NUM_REDUCE_BLOCKS,
        BLOCK_R=REDUCE_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _finalize_scalars_kernel[(C,)](
        partial_where,
        partial_mul,
        arg289_1,
        arg30_1,
        coeff_where,
        coeff_mul,
        coeff_weight,
        out0,
        NUM_REDUCE_BLOCKS_=NUM_REDUCE_BLOCKS,
        BLOCK_B=FINAL_BLOCK,
        INV_NHW_=INV_NHW,
        num_warps=4,
        num_stages=4,
    )
    _pool_epilogue_kernel[(triton.cdiv(PARENT_NUMEL, EPILOGUE_BLOCK),)](
        out1,
        mul_747,
        mul_765,
        mul_783,
        mul_801,
        mul_819,
        mul_837,
        mul_855,
        mul_873,
        mul_891,
        mul_909,
        mul_927,
        mul_945,
        arg290_1,
        full,
        getitem_315,
        arg288_1,
        arg717_1,
        coeff_where,
        coeff_mul,
        coeff_weight,
        C_=C,
        HW_=HW,
        OUT_W_=OUT_W,
        OUT_HW_=OUT_HW,
        PARENT_NUMEL_=PARENT_NUMEL,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    return out0, out1


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
