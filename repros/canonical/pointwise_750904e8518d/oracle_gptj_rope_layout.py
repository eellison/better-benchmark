"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full GPT-J RoPE pointwise/layout scope for both branches, including the expanded pair coefficients, rotate-half slice_scatter arithmetic, full_2/full_3 additive bases, tail preservation, and final transposed [4096,128] output views into one Triton kernel, whereas Inductor lowers the decomposed permute/slice/expand/clone/view/slice_scatter/add/view/permute graph as generic pointwise-layout work; Inductor cannot do this today because scheduler fusion does not form one producer/consumer group through the slice_scatter assembly and final non-contiguous layout materialization across the duplicated branches; the fix is SCHEDULER_FUSION: recognize RoPE pair-rotate plus scatter-concat layout assembly and fuse it directly into the final output strides for both branches."""
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

BATCH = 1
HEADS = 16
SEQ = 128
HEAD_DIM = 256
ROTARY_DIM = 64
ROTARY_PAIRS = ROTARY_DIM // 2
ROWS = SEQ * HEADS
HIDDEN = HEADS * HEAD_DIM
OUT_SHAPE = (HIDDEN, SEQ)
OUT_STRIDE = (1, HIDDEN)
SHAPE_PARAMS = (
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 16, 32, 2),
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 16, 32, 2),
    (1, 128, 4096),
    (1, 128, 4096),
    (128, 4096),
    (128, 4096),
)


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _validate_tensor(
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} {name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{REPRO_ID} {name} expects shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not dtype:
        raise TypeError(f"{REPRO_ID} {name} expects dtype {dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{REPRO_ID} {name} must be a CUDA tensor")
    return value


def _validate_shape_param(value: Any, expected: tuple[int, ...], name: str) -> None:
    try:
        actual = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{REPRO_ID} {name} must be a shape sequence, got {value!r}") from exc
    if actual != expected:
        raise ValueError(f"{REPRO_ID} {name} expects {expected}, got {actual}")


def _validate_inputs(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 16:
        raise ValueError(f"{REPRO_ID} expects 16 inputs, got {len(inputs)}")

    x108 = _validate_tensor(
        inputs[0],
        (BATCH, HEADS, SEQ, HEAD_DIM),
        torch.float32,
        "getitem_108",
    )
    x109 = _validate_tensor(
        inputs[1],
        (BATCH, HEADS, SEQ, HEAD_DIM),
        torch.float32,
        "getitem_109",
    )
    coeff0 = _validate_tensor(inputs[2], (BATCH, SEQ, 1, ROTARY_PAIRS, 1), torch.float32, "arg205_1")
    full2 = _validate_tensor(inputs[3], (BATCH, SEQ, HEADS, ROTARY_DIM), torch.float32, "full_2")
    coeff1 = _validate_tensor(inputs[4], (BATCH, SEQ, 1, ROTARY_PAIRS, 1), torch.float32, "arg206_1")
    full3 = _validate_tensor(inputs[5], (BATCH, SEQ, HEADS, HEAD_DIM), torch.float32, "full_3")

    devices = {x108.device, x109.device, coeff0.device, full2.device, coeff1.device, full3.device}
    if len(devices) != 1:
        raise ValueError(f"{REPRO_ID} all tensor inputs must be on the same CUDA device")

    for index, expected in enumerate(SHAPE_PARAMS, start=6):
        _validate_shape_param(inputs[index], expected, f"_shape_param_{index - 6}")

    return x108, x109, coeff0, full2, coeff1, full3


if triton is not None:

    @triton.jit
    def _gptj_rope_layout_kernel(
        x108_ptr,
        x109_ptr,
        coeff0_ptr,
        full2_ptr,
        coeff1_ptr,
        full3_ptr,
        out109_ptr,
        out108_ptr,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        coeff_s1: tl.constexpr,
        coeff_s3: tl.constexpr,
        full2_s1: tl.constexpr,
        full2_s2: tl.constexpr,
        full2_s3: tl.constexpr,
        full3_s1: tl.constexpr,
        full3_s2: tl.constexpr,
        full3_s3: tl.constexpr,
        rows_total: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        hidden: tl.constexpr,
        block_rows: tl.constexpr,
        block_d: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        dims = tl.arange(0, block_d)
        row_mask = rows < rows_total

        seq = rows // heads
        head = rows - seq * heads
        elem_mask = row_mask[:, None] & (dims[None, :] < head_dim)
        rotary = dims < rotary_dim
        rotary_mask = row_mask[:, None] & rotary[None, :]

        pair = dims // 2
        paired_dim = tl.where((dims & 1) == 0, dims + 1, dims - 1)
        is_even = (dims & 1) == 0
        rotate_sign = tl.where(is_even, 1.0, -1.0)

        x_offsets = head[:, None] * x_s1 + seq[:, None] * x_s2 + dims[None, :] * x_s3
        x_pair_offsets = head[:, None] * x_s1 + seq[:, None] * x_s2 + paired_dim[None, :] * x_s3
        full3_offsets = seq[:, None] * full3_s1 + head[:, None] * full3_s2 + dims[None, :] * full3_s3
        out_offsets = seq[:, None] * hidden + head[:, None] * head_dim + dims[None, :]

        coeff_offsets = seq[:, None] * coeff_s1 + pair[None, :] * coeff_s3
        coeff0 = tl.load(coeff0_ptr + coeff_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        coeff1 = tl.load(coeff1_ptr + coeff_offsets, mask=rotary_mask, other=0.0).to(tl.float32)

        full2_offsets = seq[:, None] * full2_s1 + head[:, None] * full2_s2 + dims[None, :] * full2_s3
        full2 = tl.load(full2_ptr + full2_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        full3 = tl.load(full3_ptr + full3_offsets, mask=elem_mask, other=0.0).to(tl.float32)

        x108 = tl.load(x108_ptr + x_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        x108_pair = tl.load(x108_ptr + x_pair_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        rot108 = x108 * coeff1 + x108_pair * coeff0 * rotate_sign[None, :]
        y108 = tl.where(rotary[None, :], full3 + full2 + rot108, full3 + x108)
        tl.store(out108_ptr + out_offsets, y108, mask=elem_mask)

        x109 = tl.load(x109_ptr + x_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        x109_pair = tl.load(x109_ptr + x_pair_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        rot109 = x109 * coeff1 + x109_pair * coeff0 * rotate_sign[None, :]
        y109 = tl.where(rotary[None, :], full3 + full2 + rot109, full3 + x109)
        tl.store(out109_ptr + out_offsets, y109, mask=elem_mask)


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact full Repro.forward scope with one fused Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    x108, x109, coeff0, full2, coeff1, full3 = _validate_inputs(tuple(inputs))
    out109 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x109.device, dtype=torch.float32)
    out108 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x108.device, dtype=torch.float32)

    block_rows = 4
    block_d = HEAD_DIM
    _gptj_rope_layout_kernel[(triton.cdiv(ROWS, block_rows),)](
        x108,
        x109,
        coeff0,
        full2,
        coeff1,
        full3,
        out109,
        out108,
        x_s0=x108.stride(0),
        x_s1=x108.stride(1),
        x_s2=x108.stride(2),
        x_s3=x108.stride(3),
        coeff_s1=coeff0.stride(1),
        coeff_s3=coeff0.stride(3),
        full2_s1=full2.stride(1),
        full2_s2=full2.stride(2),
        full2_s3=full2.stride(3),
        full3_s1=full3.stride(1),
        full3_s2=full3.stride(2),
        full3_s3=full3.stride(3),
        rows_total=ROWS,
        heads=HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        hidden=HIDDEN,
        block_rows=block_rows,
        block_d=block_d,
        num_warps=8,
        num_stages=3,
    )
    return (out109, out108)


def _check_layout(outputs: tuple[torch.Tensor, torch.Tensor]) -> bool:
    ok = True
    for index, output in enumerate(outputs):
        layout_ok = (
            tuple(output.shape) == OUT_SHAPE
            and output.stride() == OUT_STRIDE
            and output.dtype is torch.float32
            and output.storage_offset() == 0
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(output.shape)} stride={output.stride()} dtype={output.dtype})"
        )
        ok = ok and layout_ok
    return ok


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
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        ok = ok and _check_layout(layout_out)
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
