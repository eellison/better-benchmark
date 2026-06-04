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
import math
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


def run_check(inputs, *, variant: str, rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    instance = get_repro_instance()

    with torch.no_grad():
        expected = _normalize_outputs(instance(*inputs))
        actual = _normalize_outputs(oracle_forward(inputs, variant=variant))
        device = _get_device(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual)} outputs, "
            f"eager produces {len(expected)}"
        )
        return False

    all_pass = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        output_ok = True
        if got.shape != ref.shape:
            print(
                f"  output {idx}: SCOPE_MISMATCH shape oracle={list(got.shape)} "
                f"eager={list(ref.shape)}"
            )
            output_ok = False
        if got.stride() != ref.stride():
            print(
                f"  output {idx}: SCOPE_MISMATCH stride oracle={list(got.stride())} "
                f"eager={list(ref.stride())}"
            )
            output_ok = False
        if got.dtype != ref.dtype:
            print(f"  output {idx}: dtype mismatch oracle={got.dtype} eager={ref.dtype}")
            output_ok = False

        got_f32 = got.float()
        ref_f32 = ref.float()
        max_abs = (got_f32 - ref_f32).abs().max().item()
        max_rel = ((got_f32 - ref_f32).abs() / (ref_f32.abs() + 1e-8)).max().item()
        close = torch.allclose(got_f32, ref_f32, rtol=rtol, atol=atol)
        output_ok = output_ok and close
        print(
            f"  output {idx}: {'PASS' if output_ok else 'FAIL'} "
            f"(shape={list(got.shape)} dtype={got.dtype} stride={list(got.stride())} "
            f"max_abs={max_abs:.3e} max_rel={max_rel:.3e})"
        )
        all_pass = all_pass and output_ok

    return all_pass


def run_bench(
    inputs,
    *,
    variant: str,
    warmup: int = 10,
    rep: int = 50,
    no_compile: bool = False,
) -> dict[str, object]:
    device = _get_device(inputs)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Triton oracle benchmarking")

    with torch.no_grad():
        oracle_forward(inputs, variant=variant)
        torch.cuda.synchronize()
        oracle_us = _do_bench(
            lambda: oracle_forward(inputs, variant=variant),
            device,
            warmup=warmup,
            rep=rep,
        )

    compile_results: dict[str, float] = {}
    if not no_compile:
        for label, config in COMPILE_CONFIGS:
            instance = get_repro_instance()
            compiled = _compile_with_config(instance, inputs, config)
            with torch.no_grad():
                compile_results[label] = _do_bench(
                    lambda: compiled(*inputs),
                    device,
                    warmup=warmup,
                    rep=rep,
                )

    required_best = min(compile_results.values()) if compile_results else math.inf
    gate_us = min(required_best, HISTORICAL_BEST_COMPILE_US)
    valid_floor = oracle_us < gate_us
    result = {
        "repro_id": REPRO_ID,
        "variant": variant,
        "classification": CLASSIFICATION,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(gate_us, 3),
        "ratio": round(gate_us / oracle_us, 3) if oracle_us > 0 else 0.0,
        "status": "GOOD" if valid_floor else "BAD_ORACLE",
        "valid_floor": valid_floor,
        "main_oracle_path": str(REPRO_PATH.with_name(Path(__file__).name)) if valid_floor else "",
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "compile_configs_us": {k: round(v, 3) for k, v in compile_results.items()},
    }
    print(json.dumps(result))
    return result


def _compile_with_config(model: torch.nn.Module, inputs, config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config), torch.no_grad():
        compiled = torch.compile(model)
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _normalize_outputs(out) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _get_device(inputs) -> torch.device:
    for item in inputs:
        if isinstance(item, torch.Tensor):
            return item.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _do_bench(fn, device: torch.device, *, warmup: int, rep: int) -> float:
    if triton is not None and device.type == "cuda":
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    best_us = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        best_us = min(best_us, (time.perf_counter() - start) * 1_000_000.0)
    return best_us


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=50, help="Benchmark repetitions")
    parser.add_argument(
        "--variant",
        choices=(
            "single",
            "split",
            "looped",
            "looped1024",
            "looped2048",
            "looped4096",
            "looped8192",
        ),
        default=DEFAULT_VARIANT,
    )
    parser.add_argument("--no-compile", action="store_true", help="Only benchmark the Triton oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()

    if args.check:
        print(f"Checking {REPRO_ID} ({args.variant})...")
        ok = run_check(inputs, variant=args.variant, rtol=args.rtol, atol=args.atol)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID} ({args.variant})...")
        result = run_bench(
            inputs,
            variant=args.variant,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )
        if result["status"] == "BAD_ORACLE":
            print(
                "WARNING: oracle is not faster than the required compile configs "
                "and historical best compile timing; keep this artifact diagnosis-only."
            )


if __name__ == "__main__":
    main()
