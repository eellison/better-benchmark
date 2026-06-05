"""Full-scope diagnostic Triton oracle for sum_sum_342b53f3f607.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle consumes the same
five original inputs as repro.py, treats the clone/copy/slice as the live
channels 80:160 of the first input, computes both channel reductions
(`sum(slice)` and `sum(slice * (arg469 - arg479))`), and writes the returned
contiguous `[512, 80, 7, 7]` dependent BN-backward tensor plus `[80]` vector.
It differs from Inductor by forcing full-scope sibling-reduction fusion through
hand-written Triton split/single-channel schedules, but those schedules remain
slower than the 18.144 us historical best compile time. Inductor cannot use
this artifact as a true floor because the required computation is already
dominated by reading the two full tensors and writing the full output, and the
current tuned generated schedules are at least as fast as the attempted
full-scope fusion. The practical Inductor change is no new optimization from
this repro; a future fix would need to demonstrate a faster full-scope schedule
than both required local configs and the historical best.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_ID = "sum_sum_342b53f3f607"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
IN_C = 160
C = 80
H = 7
W = 7
HW = H * W
TOTAL_K = N * HW
OUT_NUMEL = N * C * HW
REDUCE_SCALE = 3.985969387755102e-05
HISTORICAL_BEST_COMPILE_US = 18.144000321626663
CLASSIFICATION = "BANDWIDTH_BOUND"
DEFAULT_VARIANT = "split"

SINGLE_BLOCK_K = 32768
SPLIT_BLOCK_K = 1024
SPLIT_N_TILES = (TOTAL_K + SPLIT_BLOCK_K - 1) // SPLIT_BLOCK_K
SPLIT_PARTIAL_BLOCK = triton.next_power_of_2(SPLIT_N_TILES) if triton is not None else 32

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


def _load_repro_module():
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs():
    module = _load_repro_module()
    return tuple(module.make_inputs())


def get_repro_instance():
    return _load_repro_module().Repro()


if triton is not None:

    @triton.jit
    def _single_channel_full_scope_kernel(
        getitem_ptr,
        arg469_ptr,
        arg479_ptr,
        arg470_ptr,
        arg190_ptr,
        out_tensor_ptr,
        out_vector_ptr,
        TOTAL_K_: tl.constexpr,
        IN_C_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0)
        k = tl.arange(0, BLOCK_K)
        mask = k < TOTAL_K_

        n = k // HW_
        hw = k - n * HW_
        source_offsets = n * (IN_C_ * HW_) + (c + C_) * HW_ + hw
        output_offsets = n * (C_ * HW_) + c * HW_ + hw

        slice_value = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        centered = (
            tl.load(arg469_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg479_ptr + c).to(tl.float32)
        )
        slice_value = tl.where(mask, slice_value, 0.0)
        centered = tl.where(mask, centered, 0.0)

        sum_slice = tl.sum(slice_value, axis=0)
        sum_slice_centered = tl.sum(slice_value * centered, axis=0)

        invstd = tl.load(arg470_ptr + c).to(tl.float32)
        grad_scale = tl.load(arg190_ptr + c).to(tl.float32)
        mean_term = sum_slice * REDUCE_SCALE_
        variance_term = sum_slice_centered * REDUCE_SCALE_ * invstd * invstd
        output_scale = invstd * grad_scale

        result = (slice_value - centered * variance_term - mean_term) * output_scale
        tl.store(out_tensor_ptr + output_offsets, result, mask=mask)
        tl.store(out_vector_ptr + c, sum_slice_centered * invstd)


    @triton.jit
    def _split_reduce_kernel(
        getitem_ptr,
        arg469_ptr,
        arg479_ptr,
        partial_sum0_ptr,
        partial_sum1_ptr,
        TOTAL_K_: tl.constexpr,
        IN_C_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        N_TILES_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < TOTAL_K_

        n = k // HW_
        hw = k - n * HW_
        source_offsets = n * (IN_C_ * HW_) + (c + C_) * HW_ + hw
        output_offsets = n * (C_ * HW_) + c * HW_ + hw

        slice_value = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        centered = (
            tl.load(arg469_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg479_ptr + c).to(tl.float32)
        )
        slice_value = tl.where(mask, slice_value, 0.0)
        centered = tl.where(mask, centered, 0.0)

        partial_offset = c * N_TILES_ + tile
        tl.store(partial_sum0_ptr + partial_offset, tl.sum(slice_value, axis=0))
        tl.store(partial_sum1_ptr + partial_offset, tl.sum(slice_value * centered, axis=0))


    @triton.jit
    def _looped_channel_full_scope_kernel(
        getitem_ptr,
        arg469_ptr,
        arg479_ptr,
        arg470_ptr,
        arg190_ptr,
        out_tensor_ptr,
        out_vector_ptr,
        TOTAL_K_: tl.constexpr,
        IN_C_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mean = tl.load(arg479_ptr + c).to(tl.float32)

        sum_slice = tl.full((), 0.0, tl.float32)
        sum_slice_centered = tl.full((), 0.0, tl.float32)

        for tile in tl.static_range(0, NUM_K_BLOCKS):
            k = tile * BLOCK_K + offsets
            mask = k < TOTAL_K_
            n = k // HW_
            hw = k - n * HW_
            source_offsets = n * (IN_C_ * HW_) + (c + C_) * HW_ + hw
            output_offsets = n * (C_ * HW_) + c * HW_ + hw

            slice_value = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
            centered = tl.load(arg469_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32) - mean
            slice_value = tl.where(mask, slice_value, 0.0)
            centered = tl.where(mask, centered, 0.0)

            sum_slice += tl.sum(slice_value, axis=0)
            sum_slice_centered += tl.sum(slice_value * centered, axis=0)

        invstd = tl.load(arg470_ptr + c).to(tl.float32)
        grad_scale = tl.load(arg190_ptr + c).to(tl.float32)
        mean_term = sum_slice * REDUCE_SCALE_
        variance_term = sum_slice_centered * REDUCE_SCALE_ * invstd * invstd
        output_scale = invstd * grad_scale
        tl.store(out_vector_ptr + c, sum_slice_centered * invstd)

        for tile in tl.static_range(0, NUM_K_BLOCKS):
            k = tile * BLOCK_K + offsets
            mask = k < TOTAL_K_
            n = k // HW_
            hw = k - n * HW_
            source_offsets = n * (IN_C_ * HW_) + (c + C_) * HW_ + hw
            output_offsets = n * (C_ * HW_) + c * HW_ + hw

            slice_value = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
            centered = tl.load(arg469_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32) - mean
            result = (slice_value - centered * variance_term - mean_term) * output_scale
            tl.store(out_tensor_ptr + output_offsets, result, mask=mask)


    @triton.jit
    def _init_cooperative_scratch_kernel(
        sum0_ptr,
        sum1_ptr,
        count_ptr,
        C_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offsets < C_
        tl.store(sum0_ptr + offsets, tl.zeros([BLOCK_C], dtype=tl.float32), mask=mask)
        tl.store(sum1_ptr + offsets, tl.zeros([BLOCK_C], dtype=tl.float32), mask=mask)
        tl.store(count_ptr + offsets, tl.zeros([BLOCK_C], dtype=tl.int32), mask=mask)


    @triton.jit
    def _cooperative_split_full_scope_kernel(
        getitem_ptr,
        arg469_ptr,
        arg479_ptr,
        arg470_ptr,
        arg190_ptr,
        sum0_ptr,
        sum1_ptr,
        count_ptr,
        out_tensor_ptr,
        out_vector_ptr,
        TOTAL_K_: tl.constexpr,
        IN_C_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        N_TILES_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < TOTAL_K_

        n = k // HW_
        hw = k - n * HW_
        source_offsets = n * (IN_C_ * HW_) + (c + C_) * HW_ + hw
        output_offsets = n * (C_ * HW_) + c * HW_ + hw

        mean = tl.load(arg479_ptr + c).to(tl.float32)
        slice_value = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        centered = tl.load(arg469_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32) - mean
        slice_value = tl.where(mask, slice_value, 0.0)
        centered = tl.where(mask, centered, 0.0)

        tl.atomic_add(sum0_ptr + c, tl.sum(slice_value, axis=0), sem="release")
        tl.atomic_add(sum1_ptr + c, tl.sum(slice_value * centered, axis=0), sem="release")
        tl.atomic_add(count_ptr + c, 1, sem="release")

        while tl.load(count_ptr + c, volatile=True) < N_TILES_:
            pass

        sum_slice = tl.load(sum0_ptr + c, volatile=True).to(tl.float32)
        sum_slice_centered = tl.load(sum1_ptr + c, volatile=True).to(tl.float32)
        invstd = tl.load(arg470_ptr + c).to(tl.float32)
        grad_scale = tl.load(arg190_ptr + c).to(tl.float32)
        mean_term = sum_slice * REDUCE_SCALE_
        variance_term = sum_slice_centered * REDUCE_SCALE_ * invstd * invstd
        output_scale = invstd * grad_scale

        if tile == 0:
            tl.store(out_vector_ptr + c, sum_slice_centered * invstd)

        slice_value_2 = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        centered_2 = tl.load(arg469_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32) - mean
        result = (slice_value_2 - centered_2 * variance_term - mean_term) * output_scale
        tl.store(out_tensor_ptr + output_offsets, result, mask=mask)


    @triton.jit
    def _split_finalize_epilogue_kernel(
        getitem_ptr,
        arg469_ptr,
        arg479_ptr,
        arg470_ptr,
        arg190_ptr,
        partial_sum0_ptr,
        partial_sum1_ptr,
        out_tensor_ptr,
        out_vector_ptr,
        TOTAL_K_: tl.constexpr,
        IN_C_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        N_TILES_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PARTIAL_BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)

        partial_offsets = tl.arange(0, PARTIAL_BLOCK)
        partial_mask = partial_offsets < N_TILES_
        partial_base = c * N_TILES_
        sum_slice = tl.sum(
            tl.load(partial_sum0_ptr + partial_base + partial_offsets, mask=partial_mask, other=0.0),
            axis=0,
        )
        sum_slice_centered = tl.sum(
            tl.load(partial_sum1_ptr + partial_base + partial_offsets, mask=partial_mask, other=0.0),
            axis=0,
        )

        invstd = tl.load(arg470_ptr + c).to(tl.float32)
        grad_scale = tl.load(arg190_ptr + c).to(tl.float32)
        mean_term = sum_slice * REDUCE_SCALE_
        variance_term = sum_slice_centered * REDUCE_SCALE_ * invstd * invstd
        output_scale = invstd * grad_scale

        if tile == 0:
            tl.store(out_vector_ptr + c, sum_slice_centered * invstd)

        k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < TOTAL_K_
        n = k // HW_
        hw = k - n * HW_
        source_offsets = n * (IN_C_ * HW_) + (c + C_) * HW_ + hw
        output_offsets = n * (C_ * HW_) + c * HW_ + hw

        slice_value = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        centered = (
            tl.load(arg469_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg479_ptr + c).to(tl.float32)
        )
        result = (slice_value - centered * variance_term - mean_term) * output_scale
        tl.store(out_tensor_ptr + output_offsets, result, mask=mask)


def _assert_inputs(inputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    getitem_3, arg469_1, arg479_1, arg470_1, arg190_1 = inputs
    assert getitem_3.shape == (N, IN_C, H, W)
    assert arg469_1.shape == (N, C, H, W)
    assert arg479_1.shape == (1, C, 1, 1)
    assert arg470_1.shape == (C,)
    assert arg190_1.shape == (C,)
    assert getitem_3.is_contiguous()
    assert arg469_1.is_contiguous()
    assert arg479_1.is_contiguous()
    assert arg470_1.is_contiguous()
    assert arg190_1.is_contiguous()
    return getitem_3, arg469_1, arg479_1, arg470_1, arg190_1


def oracle_forward(inputs, *, variant: str = DEFAULT_VARIANT):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    getitem_3, arg469_1, arg479_1, arg470_1, arg190_1 = _assert_inputs(inputs)
    out_tensor = torch.empty_like(arg469_1)
    out_vector = torch.empty_like(arg470_1)

    if variant == "single":
        _single_channel_full_scope_kernel[(C,)](
            getitem_3,
            arg469_1,
            arg479_1,
            arg470_1,
            arg190_1,
            out_tensor,
            out_vector,
            TOTAL_K_=TOTAL_K,
            IN_C_=IN_C,
            C_=C,
            HW_=HW,
            REDUCE_SCALE_=REDUCE_SCALE,
            BLOCK_K=SINGLE_BLOCK_K,
            num_warps=16,
        )
    elif variant.startswith("looped"):
        if variant == "looped":
            block_k = 2048
        else:
            block_k = int(variant.removeprefix("looped"))
        num_k_blocks = (TOTAL_K + block_k - 1) // block_k
        _looped_channel_full_scope_kernel[(C,)](
            getitem_3,
            arg469_1,
            arg479_1,
            arg470_1,
            arg190_1,
            out_tensor,
            out_vector,
            TOTAL_K_=TOTAL_K,
            IN_C_=IN_C,
            C_=C,
            HW_=HW,
            REDUCE_SCALE_=REDUCE_SCALE,
            BLOCK_K=block_k,
            NUM_K_BLOCKS=num_k_blocks,
            num_warps=8,
        )
    elif variant == "split":
        partial_sum0 = torch.empty((C, SPLIT_N_TILES), device=arg469_1.device, dtype=torch.float32)
        partial_sum1 = torch.empty((C, SPLIT_N_TILES), device=arg469_1.device, dtype=torch.float32)
        _split_reduce_kernel[(C, SPLIT_N_TILES)](
            getitem_3,
            arg469_1,
            arg479_1,
            partial_sum0,
            partial_sum1,
            TOTAL_K_=TOTAL_K,
            IN_C_=IN_C,
            C_=C,
            HW_=HW,
            N_TILES_=SPLIT_N_TILES,
            BLOCK_K=SPLIT_BLOCK_K,
            num_warps=1,
        )
        _split_finalize_epilogue_kernel[(C, SPLIT_N_TILES)](
            getitem_3,
            arg469_1,
            arg479_1,
            arg470_1,
            arg190_1,
            partial_sum0,
            partial_sum1,
            out_tensor,
            out_vector,
            TOTAL_K_=TOTAL_K,
            IN_C_=IN_C,
            C_=C,
            HW_=HW,
            N_TILES_=SPLIT_N_TILES,
            REDUCE_SCALE_=REDUCE_SCALE,
            BLOCK_K=SPLIT_BLOCK_K,
            PARTIAL_BLOCK=SPLIT_PARTIAL_BLOCK,
            num_warps=1,
        )
    else:
        raise ValueError(f"unknown oracle variant: {variant}")

    return out_tensor, out_vector


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
