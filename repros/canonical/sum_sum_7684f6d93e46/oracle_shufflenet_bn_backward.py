"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet channel-shuffle plus BatchNorm-backward-style scope, deriving the odd shuffled 58-channel slice directly from the two `[512,116,28,28]` add inputs, sharing the `sum(where)` and `sum(where * centered)` accumulators, and writing both the dense gradient epilogue and vector output without materializing the shuffle or sibling reduction intermediates; Inductor currently schedules the add/view/permute/clone/slice producer and the two dependent reductions/epilogue as generic regions with extra layout traffic; Inductor cannot do this today because the scheduler does not keep this fixed channel-shuffle producer virtual across the full multi-output reduction and its dependent epilogue; the fix is SCHEDULER_FUSION: teach reduction scheduling to inline static ShuffleNet channel-shuffle producers and plan the paired per-channel accumulators plus dense BN-backward epilogue as one full-scope template."""
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


N = 512
C = 58
VIEW_C = 116
H = 28
W = 28
HW = H * W
NHW = N * HW
INV_NHW = 2.4912308673469386e-06

SHAPE_VIEW = (N, VIEW_C, H, W)
STRIDE_VIEW = (VIEW_C * HW, HW, W, 1)
SHAPE_4D = (N, C, H, W)
STRIDE_4D = (C * HW, HW, W, 1)
SHAPE_MEAN = (1, C, 1, 1)
STRIDE_MEAN = (C, 1, 1, 1)
SHAPE_VEC = (C,)
STRIDE_VEC = (1,)
SHAPE_SHUFFLE = (N, C, 2, H, W)
SHAPE_FINAL = (N, VIEW_C, H, W)

REDUCE_BLOCK = 1024
EPILOGUE_BLOCK = 512


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _partial_sums_kernel(
        getitem_114_ptr,
        getitem_120_ptr,
        arg182_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partials_ptr,
        C_: tl.constexpr,
        VIEW_C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < NHW_
        n = k // HW_
        hw = k - n * HW_

        base = n * C_ * HW_ + c * HW_ + hw
        shuffle_channel = c * 2 + 1
        shuffle_base = n * VIEW_C_ * HW_ + shuffle_channel * HW_ + hw

        x0 = tl.load(getitem_114_ptr + shuffle_base, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(getitem_120_ptr + shuffle_base, mask=active, other=0.0).to(tl.float32)
        shuffled = x0 + x1

        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        grad = tl.load(arg182_ptr + base, mask=active, other=0.0).to(tl.float32)
        centered = grad - mean
        affine = centered * invstd
        affine = affine * weight + bias
        relu_value = tl.where(affine < 0.0, 0.0, affine)
        where_value = tl.where(relu_value <= 0.0, full_value, shuffled)
        where_value = tl.where(active, where_value, 0.0)
        centered = tl.where(active, centered, 0.0)

        partial_offset = c * NUM_TILES_ + tile
        plane = C_ * NUM_TILES_
        tl.store(partials_ptr + partial_offset, tl.sum(where_value, axis=0))
        tl.store(partials_ptr + plane + partial_offset, tl.sum(where_value * centered, axis=0))

    @triton.jit
    def _finalize_kernel(
        partials_ptr,
        invstd_ptr,
        stats_ptr,
        out_vec_ptr,
        C_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        INV_NHW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles < NUM_TILES_
        offsets = c * NUM_TILES_ + tiles
        plane = C_ * NUM_TILES_

        sum_where = tl.sum(tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum_dot = tl.sum(tl.load(partials_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        invstd_sq = invstd * invstd

        tl.store(stats_ptr + c, sum_where * INV_NHW_)
        tl.store(stats_ptr + C_ + c, (sum_dot * INV_NHW_) * invstd_sq)
        tl.store(out_vec_ptr + c, sum_dot * invstd)

    @triton.jit
    def _epilogue_kernel(
        getitem_114_ptr,
        getitem_120_ptr,
        arg182_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        stats_ptr,
        out_ptr,
        C_: tl.constexpr,
        VIEW_C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < NHW_
        n = k // HW_
        hw = k - n * HW_

        base = n * C_ * HW_ + c * HW_ + hw
        shuffle_channel = c * 2 + 1
        shuffle_base = n * VIEW_C_ * HW_ + shuffle_channel * HW_ + hw

        x0 = tl.load(getitem_114_ptr + shuffle_base, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(getitem_120_ptr + shuffle_base, mask=active, other=0.0).to(tl.float32)
        shuffled = x0 + x1

        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        grad = tl.load(arg182_ptr + base, mask=active, other=0.0).to(tl.float32)
        centered = grad - mean
        affine = centered * invstd
        affine = affine * weight + bias
        relu_value = tl.where(affine < 0.0, 0.0, affine)
        where_value = tl.where(relu_value <= 0.0, full_value, shuffled)

        mean_term = tl.load(stats_ptr + c).to(tl.float32)
        coeff = tl.load(stats_ptr + C_ + c).to(tl.float32)
        scale = invstd * weight
        out = ((where_value - centered * coeff) - mean_term) * scale
        tl.store(out_ptr + base, out, mask=active)


def _oracle_forward_torch(inputs: list[Any] | tuple[Any, ...]):
    (
        getitem_114,
        getitem_120,
        arg182_1,
        arg183_1,
        arg184_1,
        arg35_1,
        arg36_1,
        full,
        shape_param_0,
        shape_param_1,
    ) = inputs

    add_tensor = getitem_114 + getitem_120
    shuffled = add_tensor.view(shape_param_0).permute(0, 2, 1, 3, 4).clone(memory_format=torch.contiguous_format).view(shape_param_1)
    slice_tensor = shuffled[:, C:VIEW_C, :, :]

    mean = arg183_1.squeeze((0, 2, 3))[None, :, None, None]
    invstd = arg184_1.squeeze((0, 2, 3))[None, :, None, None]
    weight = arg35_1[None, :, None, None]
    bias = arg36_1[None, :, None, None]
    centered = arg182_1 - mean
    affine = centered * invstd
    affine = affine * weight + bias
    where_value = torch.where(torch.relu(affine) <= 0, full, slice_tensor)
    sum_where = where_value.sum([0, 2, 3])
    sum_dot = (where_value * centered).sum([0, 2, 3])

    mean_term = (sum_where * INV_NHW)[None, :, None, None]
    coeff = ((sum_dot * INV_NHW) * arg184_1.squeeze((0, 2, 3)).square())[None, :, None, None]
    scale = (arg184_1.squeeze((0, 2, 3)) * arg35_1)[None, :, None, None]
    out = ((where_value - centered * coeff) - mean_term) * scale
    out_vec = sum_dot * arg184_1.squeeze((0, 2, 3))
    return out, out_vec


def _require_f32_tensor(name: str, value: Any, shape: tuple[int, ...], stride: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        getitem_114,
        getitem_120,
        arg182_1,
        arg183_1,
        arg184_1,
        arg35_1,
        arg36_1,
        full,
        shape_param_0,
        shape_param_1,
    ) = inputs

    getitem_114 = _require_f32_tensor("getitem_114", getitem_114, SHAPE_VIEW, STRIDE_VIEW)
    getitem_120 = _require_f32_tensor("getitem_120", getitem_120, SHAPE_VIEW, STRIDE_VIEW)
    arg182_1 = _require_f32_tensor("arg182_1", arg182_1, SHAPE_4D, STRIDE_4D)
    arg183_1 = _require_f32_tensor("arg183_1", arg183_1, SHAPE_MEAN, STRIDE_MEAN)
    arg184_1 = _require_f32_tensor("arg184_1", arg184_1, SHAPE_MEAN, STRIDE_MEAN)
    arg35_1 = _require_f32_tensor("arg35_1", arg35_1, SHAPE_VEC, STRIDE_VEC)
    arg36_1 = _require_f32_tensor("arg36_1", arg36_1, SHAPE_VEC, STRIDE_VEC)
    full = _require_f32_tensor("full", full, (), ())
    _require_shape("_shape_param_0", shape_param_0, SHAPE_SHUFFLE)
    _require_shape("_shape_param_1", shape_param_1, SHAPE_FINAL)

    device = getitem_114.device
    if any(t.device != device for t in (getitem_120, arg182_1, arg183_1, arg184_1, arg35_1, arg36_1, full)):
        raise ValueError("all tensor inputs must be on the same device")

    return (
        getitem_114,
        getitem_120,
        arg182_1,
        arg183_1,
        arg184_1,
        arg35_1,
        arg36_1,
        full,
    )


@oracle_impl(hardware="H100", shapes="(T([512, 116, 28, 28], f32), T([512, 116, 28, 28], f32), T([512, 58, 28, 28], f32), T([1, 58, 1, 1], f32), T([1, 58, 1, 1], f32), T([58], f32), T([58], f32), T([], f32), S([512, 58, 2, 28, 28]), S([512, 116, 28, 28]))")
def oracle_forward(inputs):
    """Run the full repro-equivalent oracle computation."""
    if not isinstance(inputs, tuple):
        inputs = tuple(inputs)
    if not isinstance(inputs[0], torch.Tensor) or not inputs[0].is_cuda:
        return _oracle_forward_torch(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for this CUDA oracle")

    (
        getitem_114,
        getitem_120,
        arg182_1,
        arg183_1,
        arg184_1,
        arg35_1,
        arg36_1,
        full,
    ) = _validate_inputs(inputs)

    device = getitem_114.device
    num_tiles = triton.cdiv(NHW, REDUCE_BLOCK)
    block_tiles = triton.next_power_of_2(num_tiles)
    partials = torch.empty_strided((2, C, num_tiles), (C * num_tiles, num_tiles, 1), device=device, dtype=torch.float32)
    stats = torch.empty_strided((2, C), (C, 1), device=device, dtype=torch.float32)
    out = torch.empty_strided(SHAPE_4D, STRIDE_4D, device=device, dtype=torch.float32)
    out_vec = torch.empty_strided(SHAPE_VEC, STRIDE_VEC, device=device, dtype=torch.float32)

    _partial_sums_kernel[(C, num_tiles)](
        getitem_114,
        getitem_120,
        arg182_1,
        arg183_1,
        arg184_1,
        arg35_1,
        arg36_1,
        full,
        partials,
        C_=C,
        VIEW_C_=VIEW_C,
        HW_=HW,
        NHW_=NHW,
        NUM_TILES_=num_tiles,
        BLOCK=REDUCE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _finalize_kernel[(C,)](
        partials,
        arg184_1,
        stats,
        out_vec,
        C_=C,
        NUM_TILES_=num_tiles,
        BLOCK_TILES=block_tiles,
        INV_NHW_=INV_NHW,
        num_warps=8,
        num_stages=4,
    )
    _epilogue_kernel[(C, triton.cdiv(NHW, EPILOGUE_BLOCK))](
        getitem_114,
        getitem_120,
        arg182_1,
        arg183_1,
        arg184_1,
        arg35_1,
        arg36_1,
        full,
        stats,
        out,
        C_=C,
        VIEW_C_=VIEW_C,
        HW_=HW,
        NHW_=NHW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return out, out_vec


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
