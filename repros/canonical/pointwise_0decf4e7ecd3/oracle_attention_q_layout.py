"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle recognizes the attention-Q view/permute/cast/permute/scale/expand/view chain as a storage-linear fp16-to-fp32 scaled materialization and fills the final strided [12,64,512] backing storage with one flat Triton loop, whereas Inductor currently emits one generic two-dimensional pointwise Triton kernel over logical [512,768] indices with the same y + 768*x load/store mapping; Inductor cannot do this today because its layout algebra stops before collapsing identical input/output storage offsets into a one-dimensional storage-order loop for non-contiguous result views; the fix is ALGEBRAIC_ELIMINATION: canonicalize this layout chain to a flat storage-linear convert-and-scale materialization while preserving the final view stride."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
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

SEQ_LEN = 512
HEAD_DIM = 64
SCALE = 0.3535533905932738
BLOCK_SIZE = 1024


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


def _shape_tuple(value: object, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_68, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(addmm_68, torch.Tensor):
        raise TypeError(f"expected addmm_68 to be a tensor, got {type(addmm_68)!r}")
    if addmm_68.ndim != 2 or addmm_68.shape[0] != SEQ_LEN:
        raise ValueError(f"unexpected addmm_68 shape: {tuple(addmm_68.shape)}")
    if addmm_68.dtype is not torch.float16:
        raise ValueError(f"expected fp16 addmm_68, got {addmm_68.dtype}")
    if not addmm_68.is_cuda:
        raise ValueError("oracle_attention_q_layout.py expects CUDA inputs")

    hidden = int(addmm_68.shape[1])
    if hidden % HEAD_DIM != 0:
        raise ValueError(f"hidden dimension must be divisible by {HEAD_DIM}, got {hidden}")
    heads = hidden // HEAD_DIM
    if addmm_68.stride() != (hidden, 1):
        raise ValueError(
            f"addmm_68 must have contiguous stride ({hidden}, 1), got {addmm_68.stride()}"
        )

    expected_shape0 = (1, SEQ_LEN, hidden)
    expected_shape1 = (1, SEQ_LEN, -1, HEAD_DIM)
    expected_shape2 = (1, heads, HEAD_DIM, SEQ_LEN)
    expected_shape3 = (heads, HEAD_DIM, SEQ_LEN)
    actual_shapes = (
        _shape_tuple(shape0, "_shape_param_0"),
        _shape_tuple(shape1, "_shape_param_1"),
        _shape_tuple(shape2, "_shape_param_2"),
        _shape_tuple(shape3, "_shape_param_3"),
    )
    expected_shapes = (
        expected_shape0,
        expected_shape1,
        expected_shape2,
        expected_shape3,
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape parameters: {actual_shapes}")

    return addmm_68, hidden, heads


if triton is not None:

    @triton.jit
    def _attention_q_layout_kernel(
        input_ptr,
        output_ptr,
        total: tl.constexpr,
        block_size: tl.constexpr,
        scale: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + offsets, values * scale, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([512, 768], f16), S([1, 512, 768]), S([1, 512, -1, 64]), S([1, 12, 64, 512]), S([12, 64, 512]))")
def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    """Run the full attention-Q layout/cast/scale scope with one Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    addmm_68, hidden, heads = _validate_inputs(tuple(inputs))
    output = torch.empty_strided(
        (heads, HEAD_DIM, SEQ_LEN),
        (HEAD_DIM, 1, hidden),
        device=addmm_68.device,
        dtype=torch.float32,
    )

    total = SEQ_LEN * hidden
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    _attention_q_layout_kernel[grid](
        addmm_68,
        output,
        total=total,
        block_size=BLOCK_SIZE,
        scale=SCALE,
        num_warps=4,
        num_stages=1,
    )
    return output


def _check_layout(output: torch.Tensor, hidden: int, heads: int) -> bool:
    return (
        tuple(output.shape) == (heads, HEAD_DIM, SEQ_LEN)
        and output.stride() == (HEAD_DIM, 1, hidden)
        and output.dtype is torch.float32
        and output.storage_offset() == 0
    )


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
        addmm_68, hidden, heads = _validate_inputs(inputs)
        del addmm_68
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out, hidden, heads)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()} "
            f"dtype={layout_out.dtype})"
        )
        ok = ok and layout_ok
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
