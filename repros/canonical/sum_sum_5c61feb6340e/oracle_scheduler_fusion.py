"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet captured scope returned by Repro.forward, including both channel reductions, the `[288]` side output, and the `[64,32,14,14]` slice-add epilogue fed by twenty-three branch tensors, by sharing the two per-channel sum accumulators and emitting the channel-256:288 consumer directly from those sums and fixed slice loads, whereas Inductor currently lowers the reductions and long slice-add/pointwise consumer as generic scheduled regions around materialized intermediates; Inductor cannot do this today because its scheduler does not keep sibling reduction accumulators available to a channel-sliced downstream consumer while also sinking a static multi-input slice fan-in; the fix is SCHEDULER_FUSION: extend multi-output reduction scheduling to share accumulators and fuse fixed channel-slice consumers directly into the generated epilogue/output loop."""
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
    has_stochastic_ops,
)


N = 64
CHANNELS = 288
SLICED_CHANNELS = 32
SLICE_START = 256
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
REDUCE_ELEMENTS = N * HW
OUT1_ELEMENTS = N * SLICED_CHANNELS * HW
REDUCE_BLOCK = 16384
POINTWISE_BLOCK = 256
SCALE = 7.971938775510203e-05
CLASSIFICATION = "SCHEDULER_FUSION"

if tl is not None:
    TL_CHANNELS = tl.constexpr(CHANNELS)
    TL_SLICED_CHANNELS = tl.constexpr(SLICED_CHANNELS)
    TL_SLICE_START = tl.constexpr(SLICE_START)
    TL_HW = tl.constexpr(HW)
    TL_REDUCE_ELEMENTS = tl.constexpr(REDUCE_ELEMENTS)
    TL_OUT1_ELEMENTS = tl.constexpr(OUT1_ELEMENTS)
    TL_SCALE = tl.constexpr(SCALE)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _reduce_sums_kernel(
        arg372_ptr,
        full_ptr,
        getitem_ptr,
        arg370_ptr,
        arg690_ptr,
        arg371_ptr,
        sums_ptr,
        out0_ptr,
        BLOCK_R: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_R)
        mask = offsets < TL_REDUCE_ELEMENTS
        n_idx = offsets // TL_HW
        hw_idx = offsets - n_idx * TL_HW
        base = n_idx * TL_CHANNELS * TL_HW + channel * TL_HW + hw_idx

        pred = tl.load(arg372_ptr + base, mask=mask, other=1.0).to(tl.float32) <= 0.0
        full_value = tl.load(full_ptr).to(tl.float32)
        getitem = tl.load(getitem_ptr + base, mask=mask, other=0.0).to(tl.float32)
        where_value = tl.where(pred, full_value, getitem)

        centered = (
            tl.load(arg370_ptr + base, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg690_ptr + channel).to(tl.float32)
        )
        where_value = tl.where(mask, where_value, 0.0)
        sum_where = tl.sum(where_value, axis=0)
        sum_where_centered = tl.sum(where_value * centered, axis=0)

        arg371 = tl.load(arg371_ptr + channel).to(tl.float32)
        tl.store(sums_ptr + channel, sum_where)
        tl.store(sums_ptr + TL_CHANNELS + channel, sum_where_centered)
        tl.store(out0_ptr + channel, sum_where_centered * arg371)

    @triton.jit
    def _slice_epilogue_kernel(
        mul_306_ptr,
        mul_324_ptr,
        mul_342_ptr,
        mul_360_ptr,
        mul_378_ptr,
        mul_396_ptr,
        mul_414_ptr,
        mul_432_ptr,
        mul_450_ptr,
        mul_468_ptr,
        mul_486_ptr,
        mul_504_ptr,
        mul_522_ptr,
        mul_540_ptr,
        mul_558_ptr,
        mul_576_ptr,
        mul_594_ptr,
        mul_612_ptr,
        mul_630_ptr,
        mul_648_ptr,
        mul_666_ptr,
        mul_684_ptr,
        mul_702_ptr,
        arg372_ptr,
        full_ptr,
        getitem_ptr,
        arg370_ptr,
        arg690_ptr,
        arg371_ptr,
        arg84_ptr,
        sums_ptr,
        out1_ptr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < TL_OUT1_ELEMENTS
        hw_idx = offsets % TL_HW
        channel_offset = (offsets // TL_HW) % TL_SLICED_CHANNELS
        n_idx = offsets // (TL_SLICED_CHANNELS * TL_HW)
        channel = TL_SLICE_START + channel_offset

        base_288 = n_idx * TL_CHANNELS * TL_HW + channel * TL_HW + hw_idx
        pred = tl.load(arg372_ptr + base_288, mask=mask, other=1.0).to(tl.float32) <= 0.0
        full_value = tl.load(full_ptr).to(tl.float32)
        getitem = tl.load(getitem_ptr + base_288, mask=mask, other=0.0).to(tl.float32)
        where_value = tl.where(pred, full_value, getitem)

        centered = (
            tl.load(arg370_ptr + base_288, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg690_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        )
        sum_where = tl.load(sums_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        sum_where_centered = tl.load(
            sums_ptr + TL_CHANNELS + channel, mask=mask, other=0.0
        ).to(tl.float32)
        arg371 = tl.load(arg371_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        arg84 = tl.load(arg84_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        correction = sum_where_centered * TL_SCALE * arg371 * arg371
        mean_term = sum_where * TL_SCALE
        epilogue = (where_value - centered * correction - mean_term) * (arg371 * arg84)

        slice_sum = tl.load(
            mul_306_ptr + n_idx * 1024 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_324_ptr + n_idx * 992 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_342_ptr + n_idx * 960 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_360_ptr + n_idx * 928 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_378_ptr + n_idx * 896 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_396_ptr + n_idx * 864 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_414_ptr + n_idx * 832 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_432_ptr + n_idx * 800 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_450_ptr + n_idx * 768 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_468_ptr + n_idx * 736 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_486_ptr + n_idx * 704 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_504_ptr + n_idx * 672 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_522_ptr + n_idx * 640 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_540_ptr + n_idx * 608 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_558_ptr + n_idx * 576 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_576_ptr + n_idx * 544 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_594_ptr + n_idx * 512 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_612_ptr + n_idx * 480 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_630_ptr + n_idx * 448 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_648_ptr + n_idx * 416 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_666_ptr + n_idx * 384 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_684_ptr + n_idx * 352 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        slice_sum += tl.load(
            mul_702_ptr + n_idx * 320 * TL_HW + channel * TL_HW + hw_idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        tl.store(out1_ptr + offsets, slice_sum + epilogue, mask=mask)


def _expect_f32_tensor(
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
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 30:
        raise ValueError(f"{REPRO_ID} expects 30 inputs, got {len(inputs)}")

    expected_branch_channels = (
        1024,
        992,
        960,
        928,
        896,
        864,
        832,
        800,
        768,
        736,
        704,
        672,
        640,
        608,
        576,
        544,
        512,
        480,
        448,
        416,
        384,
        352,
        320,
    )
    checked: list[torch.Tensor] = []
    for index, channels in enumerate(expected_branch_channels):
        checked.append(
            _expect_f32_tensor(
                f"branch input {index}",
                inputs[index],
                (N, channels, HEIGHT, WIDTH),
                (channels * HW, HW, WIDTH, 1),
            )
        )

    checked.append(
        _expect_f32_tensor(
            "arg372_1",
            inputs[23],
            (N, CHANNELS, HEIGHT, WIDTH),
            (CHANNELS * HW, HW, WIDTH, 1),
        )
    )
    checked.append(_expect_f32_tensor("full", inputs[24], (), ()))
    checked.append(
        _expect_f32_tensor(
            "getitem_234",
            inputs[25],
            (N, CHANNELS, HEIGHT, WIDTH),
            (CHANNELS * HW, HW, WIDTH, 1),
        )
    )
    checked.append(
        _expect_f32_tensor(
            "arg370_1",
            inputs[26],
            (N, CHANNELS, HEIGHT, WIDTH),
            (CHANNELS * HW, HW, WIDTH, 1),
        )
    )
    checked.append(
        _expect_f32_tensor(
            "arg690_1",
            inputs[27],
            (1, CHANNELS, 1, 1),
            (CHANNELS, 1, 1, 1),
        )
    )
    checked.append(_expect_f32_tensor("arg371_1", inputs[28], (CHANNELS,), (1,)))
    checked.append(_expect_f32_tensor("arg84_1", inputs[29], (CHANNELS,), (1,)))

    device = checked[0].device
    if any(t.device != device for t in checked):
        raise ValueError("all tensor inputs must be on the same device")
    return tuple(checked)


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = _validate_inputs(inputs)
    branch_inputs = tensors[:23]
    arg372, full, getitem, arg370, arg690, arg371, arg84 = tensors[23:]

    slice_sum = branch_inputs[0][:, SLICE_START:CHANNELS, :, :]
    for tensor in branch_inputs[1:]:
        slice_sum = slice_sum + tensor[:, SLICE_START:CHANNELS, :, :]

    where_value = torch.where(arg372 <= 0, full, getitem)
    sum_where = where_value.sum(dim=(0, 2, 3))
    centered = arg370 - arg690
    sum_where_centered = (where_value * centered).sum(dim=(0, 2, 3))
    out0 = sum_where_centered * arg371

    correction = (sum_where_centered * SCALE * arg371 * arg371)[None, :, None, None]
    mean_term = (sum_where * SCALE)[None, :, None, None]
    scale = (arg371 * arg84)[None, :, None, None]
    epilogue = (where_value - centered * correction - mean_term) * scale
    out1 = slice_sum + epilogue[:, SLICE_START:CHANNELS, :, :]
    return out0, out1


@oracle_impl(hardware="H100", shapes="(T([64, 1024, 14, 14], f32), T([64, 992, 14, 14], f32), T([64, 960, 14, 14], f32), T([64, 928, 14, 14], f32), T([64, 896, 14, 14], f32), T([64, 864, 14, 14], f32), T([64, 832, 14, 14], f32), T([64, 800, 14, 14], f32), T([64, 768, 14, 14], f32), T([64, 736, 14, 14], f32), T([64, 704, 14, 14], f32), T([64, 672, 14, 14], f32), T([64, 640, 14, 14], f32), T([64, 608, 14, 14], f32), T([64, 576, 14, 14], f32), T([64, 544, 14, 14], f32), T([64, 512, 14, 14], f32), T([64, 480, 14, 14], f32), T([64, 448, 14, 14], f32), T([64, 416, 14, 14], f32), T([64, 384, 14, 14], f32), T([64, 352, 14, 14], f32), T([64, 320, 14, 14], f32), T([64, 288, 14, 14], f32), T([], f32), T([64, 288, 14, 14], f32), T([64, 288, 14, 14], f32), T([1, 288, 1, 1], f32), T([288], f32), T([288], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full two-output Repro.forward computation."""
    tensors = _validate_inputs(inputs)
    if triton is None or not tensors[0].is_cuda:
        return _torch_reference(inputs)

    branch_inputs = tensors[:23]
    arg372, full, getitem, arg370, arg690, arg371, arg84 = tensors[23:]
    out0 = torch.empty_strided(
        (CHANNELS,),
        (1,),
        device=arg372.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        (N, SLICED_CHANNELS, HEIGHT, WIDTH),
        (SLICED_CHANNELS * HW, HW, WIDTH, 1),
        device=arg372.device,
        dtype=torch.float32,
    )
    sums = torch.empty_strided(
        (2, CHANNELS),
        (CHANNELS, 1),
        device=arg372.device,
        dtype=torch.float32,
    )

    _reduce_sums_kernel[(CHANNELS,)](
        arg372,
        full,
        getitem,
        arg370,
        arg690,
        arg371,
        sums,
        out0,
        BLOCK_R=REDUCE_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _slice_epilogue_kernel[(triton.cdiv(OUT1_ELEMENTS, POINTWISE_BLOCK),)](
        branch_inputs[0],
        branch_inputs[1],
        branch_inputs[2],
        branch_inputs[3],
        branch_inputs[4],
        branch_inputs[5],
        branch_inputs[6],
        branch_inputs[7],
        branch_inputs[8],
        branch_inputs[9],
        branch_inputs[10],
        branch_inputs[11],
        branch_inputs[12],
        branch_inputs[13],
        branch_inputs[14],
        branch_inputs[15],
        branch_inputs[16],
        branch_inputs[17],
        branch_inputs[18],
        branch_inputs[19],
        branch_inputs[20],
        branch_inputs[21],
        branch_inputs[22],
        arg372,
        full,
        getitem,
        arg370,
        arg690,
        arg371,
        arg84,
        sums,
        out1,
        BLOCK_M=POINTWISE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return out0, out1


# --- CLI entry point ---
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
