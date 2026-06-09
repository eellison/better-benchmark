"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle recognizes the complete attention-K layout chain as a storage-order fp16-to-fp32 cast into the permuted [1,12,512,64] stride (393216,64,768,1) buffer and returns the final [12,512,64] stride (64,768,1) view, whereas Inductor currently emits one generic pointwise kernel for fused convert_element_type/permute/view over the logical permuted index space; Inductor cannot do this today because its layout simplifier does not canonicalize view-permute-cast-expand-view chains whose cast preserves input storage order into a direct dtype-conversion copy plus as_strided result; the fix is ALGEBRAIC_ELIMINATION: add a guarded layout-chain rewrite that collapses no-op expand and view/permute metadata around dtype conversion into a storage-order cast and final as_strided alias."""
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

BATCH = 1
SEQ_LEN = 512
HEAD_DIM = 64
BLOCK_SIZE = 1024
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _storage_order_cast_kernel(
        input_ptr,
        cast_ptr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(input_ptr + offsets).to(tl.float32)
        tl.store(cast_ptr + offsets, values)


def _shape_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"{name} must be a shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[object] | tuple[object, ...]) -> tuple[torch.Tensor, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    addmm_69, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(addmm_69, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor, got {type(addmm_69)!r}")
    if addmm_69.dtype is not torch.float16:
        raise TypeError(f"{REPRO_ID} expects fp16 input, got {addmm_69.dtype}")
    if not addmm_69.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if addmm_69.ndim != 2 or addmm_69.shape[0] != SEQ_LEN:
        raise ValueError(f"{REPRO_ID} expects input shape [512, hidden], got {tuple(addmm_69.shape)}")

    hidden = int(addmm_69.shape[1])
    if hidden % HEAD_DIM != 0:
        raise ValueError(f"hidden size must be divisible by {HEAD_DIM}, got {hidden}")
    heads = hidden // HEAD_DIM
    if tuple(addmm_69.stride()) != (hidden, 1):
        raise ValueError(f"{REPRO_ID} expects contiguous [512, hidden] input, got stride={addmm_69.stride()}")
    if addmm_69.numel() % BLOCK_SIZE != 0:
        raise ValueError(f"{REPRO_ID} expects numel divisible by {BLOCK_SIZE}, got {addmm_69.numel()}")

    expected_shape0 = (BATCH, SEQ_LEN, hidden)
    expected_shape1 = (BATCH, SEQ_LEN, -1, HEAD_DIM)
    inferred_shape1 = (BATCH, SEQ_LEN, heads, HEAD_DIM)
    expected_shape2 = (BATCH, heads, SEQ_LEN, HEAD_DIM)
    expected_shape3 = (heads, SEQ_LEN, HEAD_DIM)

    got_shape0 = _shape_tuple(shape0, "_shape_param_0")
    got_shape1 = _shape_tuple(shape1, "_shape_param_1")
    got_shape2 = _shape_tuple(shape2, "_shape_param_2")
    got_shape3 = _shape_tuple(shape3, "_shape_param_3")
    if got_shape0 != expected_shape0:
        raise ValueError(f"unexpected _shape_param_0: {got_shape0}, expected {expected_shape0}")
    if got_shape1 not in (expected_shape1, inferred_shape1):
        raise ValueError(f"unexpected _shape_param_1: {got_shape1}, expected {expected_shape1}")
    if got_shape2 != expected_shape2:
        raise ValueError(f"unexpected _shape_param_2: {got_shape2}, expected {expected_shape2}")
    if got_shape3 != expected_shape3:
        raise ValueError(f"unexpected _shape_param_3: {got_shape3}, expected {expected_shape3}")

    return addmm_69, heads


def _permuted_stride(hidden: int) -> tuple[int, int, int, int]:
    return (SEQ_LEN * hidden, HEAD_DIM, hidden, 1)


def _output_stride(hidden: int) -> tuple[int, int, int]:
    return (HEAD_DIM, hidden, 1)


@oracle_impl(hardware="H100", shapes="(T([512, 768], f16), S([1, 512, 768]), S([1, 512, -1, 64]), S([1, 12, 512, 64]), S([12, 512, 64]))")
def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full attention-K layout graph while preserving eager strides."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    addmm_69, heads = _validate_inputs(inputs)
    hidden = int(addmm_69.shape[1])
    cast_buffer = torch.empty_strided(
        (BATCH, heads, SEQ_LEN, HEAD_DIM),
        _permuted_stride(hidden),
        device=addmm_69.device,
        dtype=torch.float32,
    )

    grid = (addmm_69.numel() // BLOCK_SIZE,)
    _storage_order_cast_kernel[grid](
        addmm_69,
        cast_buffer,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return torch.as_strided(
        cast_buffer,
        (heads, SEQ_LEN, HEAD_DIM),
        _output_stride(hidden),
    )


def _check_layout_and_values(
    instance: torch.nn.Module,
    inputs: list[object] | tuple[object, ...],
) -> bool:
    addmm_69, heads = _validate_inputs(inputs)
    hidden = int(addmm_69.shape[1])
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(actual.shape) == (heads, SEQ_LEN, HEAD_DIM)
        and actual.stride() == _output_stride(hidden)
        and actual.dtype is torch.float32
        and actual.storage_offset() == 0
    )
    exact_ok = torch.equal(expected, actual)
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()} dtype={actual.dtype} "
        f"storage_offset={actual.storage_offset()})"
    )
    print(f"  output 0 exact_values: {'PASS' if exact_ok else 'FAIL'}")
    return layout_ok and exact_ok


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
        ok = _check_layout_and_values(instance, inputs) and ok
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
