"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete multi-output ViT backward reduction scope, including the mm view, channel scale, sibling per-token sums, strided arg83 layout/view producer, final add update, batch reduction output, and the two channel reductions in two Triton reduction passes, whereas Inductor currently schedules the decomposed reductions and surrounding pointwise/layout views as generic fused regions that reread or materialize intermediate tensors across sibling outputs; Inductor cannot do this today because its scheduler does not form one multi-output reduction plan that shares the row reductions with the downstream batch/channel reductions while preserving the transposed arg83 view as a tiled virtual producer; the fix is SCHEDULER_FUSION: add a multi-output reduction schedule that keeps sibling row sums and layout-only producers inside the final reduction consumers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
TOKENS = 256
CHANNELS = 768
ROW_COUNT = BATCH * TOKENS
INV_CHANNELS = 1.0 / CHANNELS

ROW_BLOCK_N = 8
ROW_BLOCK_C = 64
FINAL_BLOCK_N = 8
FINAL_BLOCK_C = 32
ZERO_BLOCK = 1024


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _row_sums_kernel(
        mm_ptr,
        scale_ptr,
        arg83_ptr,
        arg2_ptr,
        arg84_ptr,
        arg85_ptr,
        row_sums_ptr,
        batch: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.program_id(0)
        n_block = tl.program_id(1)
        n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < tokens

        sum0 = tl.zeros((BLOCK_N,), tl.float32)
        sum1_unscaled = tl.zeros((BLOCK_N,), tl.float32)
        a84 = tl.load(
            arg84_ptr + b * tokens + n_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        a85 = tl.load(
            arg85_ptr + b * tokens + n_offsets,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        c_offsets = tl.arange(0, BLOCK_C)
        for c_base in tl.static_range(0, channels, BLOCK_C):
            c = c_base + c_offsets
            c_mask = c < channels

            matrix_mask = c_mask[:, None] & n_mask[None, :]
            mm_offsets = b * tokens * channels + n_offsets[None, :] * channels + c[:, None]
            arg83_offsets = b * channels * tokens + c[:, None] * tokens + n_offsets[None, :]
            arg2_offsets = n_offsets[None, :] * channels + c[:, None]

            x = tl.load(mm_ptr + mm_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
            x_scaled = x * scale[:, None]
            permuted = tl.load(
                arg83_ptr + arg83_offsets,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            bias = tl.load(arg2_ptr + arg2_offsets, mask=matrix_mask, other=0.0).to(
                tl.float32
            )

            centered = permuted + bias - a84[None, :]
            sum0 += tl.sum(x_scaled, axis=0)
            sum1_unscaled += tl.sum(x_scaled * centered, axis=0)

        row_ids = b * tokens + n_offsets
        tl.store(row_sums_ptr + row_ids, sum0, mask=n_mask)
        tl.store(row_sums_ptr + batch * tokens + row_ids, sum1_unscaled * a85, mask=n_mask)

    @triton.jit
    def _zero_channel_outputs_kernel(
        out0_ptr,
        out1_ptr,
        out3_ptr,
        channels: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < channels
        zeros = tl.zeros((BLOCK,), tl.float32)
        tl.store(out0_ptr + offsets, zeros, mask=mask)
        tl.store(out1_ptr + offsets, zeros, mask=mask)
        tl.store(out3_ptr + offsets, zeros, mask=mask)

    @triton.jit
    def _final_outputs_kernel(
        mm_ptr,
        scale_ptr,
        arg83_ptr,
        arg2_ptr,
        arg84_ptr,
        arg85_ptr,
        add_ptr,
        row_sums_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        batch: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        inv_channels: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        c_block = tl.program_id(1)
        n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        n_mask = n_offsets < tokens
        c_mask = c_offsets < channels
        matrix_mask = n_mask[:, None] & c_mask[None, :]

        scale = tl.load(scale_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        arg2 = tl.load(
            arg2_ptr + n_offsets[:, None] * channels + c_offsets[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)

        out2_acc = tl.zeros((BLOCK_N, BLOCK_C), tl.float32)
        out0_acc = tl.zeros((BLOCK_C,), tl.float32)
        out1_acc = tl.zeros((BLOCK_C,), tl.float32)
        out3_acc = tl.zeros((BLOCK_C,), tl.float32)

        for b in tl.range(0, batch):
            row_ids = b * tokens + n_offsets
            a84 = tl.load(arg84_ptr + row_ids, mask=n_mask, other=0.0).to(tl.float32)
            a85 = tl.load(arg85_ptr + row_ids, mask=n_mask, other=0.0).to(tl.float32)
            sum0 = tl.load(row_sums_ptr + row_ids, mask=n_mask, other=0.0).to(tl.float32)
            sum1 = tl.load(
                row_sums_ptr + batch * tokens + row_ids,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)

            mm_offsets = b * tokens * channels + n_offsets[:, None] * channels + c_offsets[None, :]
            arg83_offsets = b * channels * tokens + c_offsets[:, None] * tokens + n_offsets[None, :]
            add_offsets = b * tokens * channels + n_offsets[:, None] * channels + c_offsets[None, :]

            x = tl.load(mm_ptr + mm_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
            permuted_raw = tl.load(
                arg83_ptr + arg83_offsets,
                mask=c_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            permuted = tl.trans(permuted_raw)
            add_value = tl.load(add_ptr + add_offsets, mask=matrix_mask, other=0.0).to(
                tl.float32
            )

            centered = permuted + arg2 - a84[:, None]
            m2 = centered * a85[:, None]
            x_scaled = x * scale[None, :]
            updated = add_value + a85[:, None] * x_scaled - (
                a85[:, None] * inv_channels
            ) * (sum0[:, None] + m2 * sum1[:, None])

            out2_acc += updated
            out0_acc += tl.sum(x * m2, axis=0)
            out1_acc += tl.sum(x, axis=0)
            out3_acc += tl.sum(updated, axis=0)

        tl.store(
            out2_ptr + n_offsets[:, None] * channels + c_offsets[None, :],
            out2_acc,
            mask=matrix_mask,
        )
        tl.atomic_add(out0_ptr + c_offsets, out0_acc, sem="relaxed", mask=c_mask)
        tl.atomic_add(out1_ptr + c_offsets, out1_acc, sem="relaxed", mask=c_mask)
        tl.atomic_add(out3_ptr + c_offsets, out3_acc, sem="relaxed", mask=c_mask)


def _validate_inputs(inputs: tuple[Any, ...]) -> None:
    if len(inputs) != 10:
        raise ValueError(f"expected 10 inputs, got {len(inputs)}")

    (
        mm_104,
        arg3_1,
        arg83_1,
        arg2_1,
        arg84_1,
        arg85_1,
        add_49,
        shape_param_0,
        shape_param_1,
        shape_param_2,
    ) = inputs

    expected = {
        "mm_104": (mm_104, (ROW_COUNT, CHANNELS)),
        "arg3_1": (arg3_1, (CHANNELS,)),
        "arg83_1": (arg83_1, (BATCH, CHANNELS, 16, 16)),
        "arg2_1": (arg2_1, (1, TOKENS, CHANNELS)),
        "arg84_1": (arg84_1, (BATCH, TOKENS, 1)),
        "arg85_1": (arg85_1, (BATCH, TOKENS, 1)),
        "add_49": (add_49, (BATCH, TOKENS, CHANNELS)),
    }
    for name, (tensor, shape) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} expected shape={shape} dtype=torch.float32, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if list(shape_param_0) != [BATCH, TOKENS, CHANNELS]:
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0}")
    if list(shape_param_1) != [BATCH, CHANNELS, TOKENS]:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1}")
    if list(shape_param_2) != [BATCH, CHANNELS, 16, 16]:
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2}")


@oracle_impl(hardware="H100", shapes="(T([32768, 768], f32), T([768], f32), T([128, 768, 16, 16], f32), T([1, 256, 768], f32), T([128, 256, 1], f32), T([128, 256, 1], f32), T([128, 256, 768], f32), S([128, 256, 768]), S([128, 768, 256]), S([128, 768, 16, 16]))")
def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    if triton is None:
        raise RuntimeError("triton is not available")

    _validate_inputs(inputs)
    (
        mm_104,
        arg3_1,
        arg83_1,
        arg2_1,
        arg84_1,
        arg85_1,
        add_49,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    row_sums = torch.empty((2, BATCH, TOKENS), device=mm_104.device, dtype=torch.float32)
    out0 = torch.empty((CHANNELS,), device=mm_104.device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=mm_104.device, dtype=torch.float32)
    out2 = torch.empty_strided(
        (1, TOKENS, CHANNELS),
        (TOKENS * CHANNELS, CHANNELS, 1),
        device=mm_104.device,
        dtype=torch.float32,
    )
    out3 = torch.empty((CHANNELS,), device=mm_104.device, dtype=torch.float32)

    _row_sums_kernel[(BATCH, triton.cdiv(TOKENS, ROW_BLOCK_N))](
        mm_104,
        arg3_1,
        arg83_1,
        arg2_1,
        arg84_1,
        arg85_1,
        row_sums,
        BATCH,
        TOKENS,
        CHANNELS,
        BLOCK_N=ROW_BLOCK_N,
        BLOCK_C=ROW_BLOCK_C,
        num_warps=8,
    )
    _zero_channel_outputs_kernel[(triton.cdiv(CHANNELS, ZERO_BLOCK),)](
        out0,
        out1,
        out3,
        CHANNELS,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _final_outputs_kernel[
        (triton.cdiv(TOKENS, FINAL_BLOCK_N), triton.cdiv(CHANNELS, FINAL_BLOCK_C))
    ](
        mm_104,
        arg3_1,
        arg83_1,
        arg2_1,
        arg84_1,
        arg85_1,
        add_49,
        row_sums,
        out0,
        out1,
        out2,
        out3,
        BATCH,
        TOKENS,
        CHANNELS,
        INV_CHANNELS,
        BLOCK_N=FINAL_BLOCK_N,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=4,
    )

    return out0, out1, out2, out3


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify correctness against eager Repro",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark oracle vs torch.compile",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Repetitions for benchmark",
    )
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Benchmark across all shapes from shapes.txt",
    )
    parser.add_argument(
        "--show-hw",
        action="store_true",
        help="Print GPU hardware info and exit",
    )
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
