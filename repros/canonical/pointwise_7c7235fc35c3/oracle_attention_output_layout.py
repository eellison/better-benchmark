"""Gap diagnosis (classification: NEW_PATTERN): this oracle collapses the full f32[12,512,64] attention-output view/cast/permute/contiguous-clone/final-view graph into one Triton kernel that reads the head-major fp32 input and writes the final dense fp16[512,768] layout directly, whereas Inductor currently lowers this view-threaded cast-transpose materialization through its generic pointwise/layout path instead of recognizing the attention-output transpose-and-pack idiom; Inductor cannot do this today because pointwise codegen does not select a specialized head/sequence tiled cast-transpose pattern for attention outputs; the fix is NEW_PATTERN: add an attention-output layout codegen pattern that sinks the fp32-to-fp16 cast into the contiguous clone and writes the flattened final layout directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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

SEQ_LEN = 512
HEAD_DIM = 64
DEFAULT_HEADS = 12
DEFAULT_OUT_COLS = DEFAULT_HEADS * HEAD_DIM
BLOCK_M = 16
BLOCK_N = 128
NUM_WARPS = 4
CLASSIFICATION = "NEW_PATTERN"


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _attention_output_layout_kernel(
        input_ptr,
        output_ptr,
        in_stride_h: tl.constexpr,
        in_stride_s: tl.constexpr,
        in_stride_d: tl.constexpr,
        heads: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        out_cols: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        seq_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
        col_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)

        seqs = seq_offsets[:, None]
        cols = col_offsets[None, :]
        mask = (seqs < seq_len) & (cols < out_cols)

        heads_idx = cols // head_dim
        dims = cols - heads_idx * head_dim
        values = tl.load(
            input_ptr + heads_idx * in_stride_h + seqs * in_stride_s + dims * in_stride_d,
            mask=mask & (heads_idx < heads),
            other=0.0,
        )
        tl.store(output_ptr + seqs * out_cols + cols, values, mask=mask)


def _shape_tuple(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"{name} must be a shape sequence, got {type(value)!r}")


def _validate_inputs(inputs: Sequence[object]) -> tuple[torch.Tensor, int]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects tensor plus three shape params, got {len(inputs)}")

    bmm_23 = inputs[0]
    if not isinstance(bmm_23, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(bmm_23)!r}")
    if bmm_23.dtype != torch.float32:
        raise TypeError(f"input 0 must be torch.float32, got {bmm_23.dtype}")
    if not bmm_23.is_cuda:
        raise ValueError("oracle_attention_output_layout.py expects CUDA inputs")
    if bmm_23.ndim != 3:
        raise ValueError(f"input 0 must be rank 3, got shape={tuple(bmm_23.shape)}")

    heads, seq_len, head_dim = (int(dim) for dim in bmm_23.shape)
    if seq_len != SEQ_LEN or head_dim != HEAD_DIM:
        raise ValueError(
            f"unexpected input shape {tuple(bmm_23.shape)}; expected [heads, {SEQ_LEN}, {HEAD_DIM}]"
        )
    if not bmm_23.is_contiguous():
        raise ValueError(f"input 0 must be contiguous, got stride={tuple(bmm_23.stride())}")

    shape0 = _shape_tuple(inputs[1], "_shape_param_0")
    shape1 = _shape_tuple(inputs[2], "_shape_param_1")
    shape2 = _shape_tuple(inputs[3], "_shape_param_2")
    expected_shape0 = (1, heads, SEQ_LEN, HEAD_DIM)
    expected_shape1 = (1, SEQ_LEN, -1)
    expected_shape2 = (SEQ_LEN, heads * HEAD_DIM)
    if shape0 != expected_shape0:
        raise ValueError(f"unexpected _shape_param_0: {shape0}, expected {expected_shape0}")
    if shape1 != expected_shape1:
        raise ValueError(f"unexpected _shape_param_1: {shape1}, expected {expected_shape1}")
    if shape2 != expected_shape2:
        raise ValueError(f"unexpected _shape_param_2: {shape2}, expected {expected_shape2}")
    return bmm_23, heads


def oracle_forward(inputs: Sequence[object]) -> torch.Tensor:
    """Run the full Repro.forward scope and return the final dense fp16 layout."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    bmm_23, heads = _validate_inputs(inputs)
    out_cols = heads * HEAD_DIM
    output = torch.empty_strided(
        (SEQ_LEN, out_cols),
        (out_cols, 1),
        device=bmm_23.device,
        dtype=torch.float16,
    )
    grid = (triton.cdiv(SEQ_LEN, BLOCK_M), triton.cdiv(out_cols, BLOCK_N))
    _attention_output_layout_kernel[grid](
        bmm_23,
        output,
        in_stride_h=int(bmm_23.stride(0)),
        in_stride_s=int(bmm_23.stride(1)),
        in_stride_d=int(bmm_23.stride(2)),
        heads=heads,
        seq_len=SEQ_LEN,
        head_dim=HEAD_DIM,
        out_cols=out_cols,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=4,
    )
    return output


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _check_layout_and_values(
    instance: torch.nn.Module,
    inputs: Sequence[object],
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == tuple(eager_out.shape)
        and oracle_out.dtype == eager_out.dtype
        and oracle_out.stride() == eager_out.stride()
        and oracle_out.storage_offset() == eager_out.storage_offset()
    )
    values_ok = torch.allclose(oracle_out.float(), eager_out.float(), atol=atol, rtol=rtol)
    exact_ok = torch.equal(oracle_out, eager_out)
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} dtype={oracle_out.dtype} "
        f"stride={tuple(oracle_out.stride())} storage_offset={oracle_out.storage_offset()})"
    )
    print(
        f"  output 0 values: {'PASS' if values_ok else 'FAIL'} "
        f"(exact={exact_ok} max_abs={_max_abs(oracle_out, eager_out):.2e})"
    )
    return layout_ok and values_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
        ok = _check_layout_and_values(instance, inputs, args.atol, args.rtol) and ok
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
