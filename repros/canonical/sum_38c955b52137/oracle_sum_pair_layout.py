"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle writes the final transposed-view backing storage while performing the static size-2 head-pair bf16 sum in one no-mask Triton kernel, whereas Inductor currently emits a generic fused pointwise kernel over the squeezed-sum logical layout and reinterprets its contiguous clone as the returned transpose; Inductor cannot do this today because the scheduler has no dedicated layout-aware reduction template that starts from the returned layout and removes generic pointwise indexing overhead; the fix is SCHEDULER_FUSION: add a shape-specialized materializing reduction template for static split-dimension sums feeding clone/view/permute returns."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
    get_shape_key,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
IN_HEADS = 16
OUT_HEADS = 8
SEQ_LEN = 512
HEAD_DIM = 128
ROWS = BATCH * SEQ_LEN
COLS = OUT_HEADS * HEAD_DIM

INPUT_SHAPE = (BATCH, IN_HEADS, SEQ_LEN, HEAD_DIM)
VIEW_SHAPE = (BATCH, OUT_HEADS, 2, SEQ_LEN, HEAD_DIM)
MID_VIEW_SHAPE = (BATCH, SEQ_LEN, COLS)
PRE_TRANSPOSE_SHAPE = (ROWS, COLS)
OUTPUT_SHAPE = (COLS, ROWS)
OUTPUT_STRIDE = (1, COLS)


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _sum_pair_layout_kernel(
        x_ptr,
        out_ptr,
        in_s0: tl.constexpr,
        in_s1: tl.constexpr,
        in_s2: tl.constexpr,
        in_s3: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        SEQ_LEN_C: tl.constexpr,
        HEAD_DIM_C: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        col_block = tl.program_id(1)

        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

        batch = rows // SEQ_LEN_C
        seq = rows - batch * SEQ_LEN_C
        out_head = cols // HEAD_DIM_C
        dim = cols - out_head * HEAD_DIM_C

        x0_offsets = (
            batch[:, None] * in_s0
            + (out_head[None, :] * 2) * in_s1
            + seq[:, None] * in_s2
            + dim[None, :] * in_s3
        )
        x1_offsets = x0_offsets + in_s1
        values = tl.load(x_ptr + x0_offsets) + tl.load(x_ptr + x1_offsets)

        out_offsets = cols[None, :] * out_s0 + rows[:, None] * out_s1
        tl.store(out_ptr + out_offsets, values)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter to be iterable, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    x, shape_param_0, shape_param_1, shape_param_2 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected input 0 to be a tensor, got {type(x)!r}")
    if x.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 input, got {x.dtype}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={x.stride()}")
    if _shape_tuple(shape_param_0) != VIEW_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    if _shape_tuple(shape_param_1) != MID_VIEW_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1!r}")
    if _shape_tuple(shape_param_2) != PRE_TRANSPOSE_SHAPE:
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2!r}")
    return x


@oracle_impl(hardware="H100", shapes="(T([4, 16, 512, 128], bf16), S([4, 8, 2, 512, 128]), S([4, 512, 1024]), S([2048, 1024]))")
def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    """Run the full sum -> squeeze -> permute -> clone -> view -> permute scope."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    x = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=x.dtype,
    )

    block_m = 8
    block_n = HEAD_DIM
    grid = (triton.cdiv(ROWS, block_m), triton.cdiv(COLS, block_n))
    _sum_pair_layout_kernel[grid](
        x,
        output,
        in_s0=x.stride(0),
        in_s1=x.stride(1),
        in_s2=x.stride(2),
        in_s3=x.stride(3),
        out_s0=output.stride(0),
        out_s1=output.stride(1),
        SEQ_LEN_C=SEQ_LEN,
        HEAD_DIM_C=HEAD_DIM,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=3,
    )
    return output


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
