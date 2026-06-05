# Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Moondream residual LayerNorm scope in one shape-specialized Triton row kernel, including the two fp16 residual adds, fp32 raw-moment var_mean over hidden size 2048, eps=1e-5 affine epilogue with fp16 weight/bias, final fp16 cast, and four returned `[512, 2048]` views aliasing one `[1, 512, 2048]` storage, whereas Inductor emits one fused generic var_mean kernel but uses Welford reduction state and then reloads the three activation inputs for the normalize/affine pass; Inductor cannot do this today because its normalization lowering does not have a guarded correction=0 raw-moments row template that can retain the loaded tile through the epilogue within the accepted fp16 tolerance envelope; the fix is ALGEBRAIC_ELIMINATION: add a fixed-hidden LayerNorm lowering that replaces Welford with sum/sum-of-squares moments and avoids the second activation read when accuracy policy allows it.
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

ROWS = 512
HIDDEN = 2048
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (1, ROWS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
BASE_STRIDE = (ROWS * HIDDEN, HIDDEN, 1)
OUTPUT_STRIDE = (HIDDEN, 1)
BLOCK_H = 2048
EPS = 1.0e-5
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _residual_layernorm_aliases_kernel(
        addmm0_ptr,
        addmm1_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        offsets = row * hidden + cols

        addmm0 = tl.load(addmm0_ptr + offsets)
        addmm1 = tl.load(addmm1_ptr + offsets)
        residual = tl.load(residual_ptr + offsets)

        # Repro.forward performs fp16 add, then another fp16 add, before the
        # explicit fp32 conversion feeding var_mean.
        summed = (addmm0 + addmm1).to(tl.float16)
        x = (summed + residual).to(tl.float16).to(tl.float32)

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y.to(tl.float16))


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    (
        addmm_135,
        addmm_137,
        add_178,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    tensor_inputs = (addmm_135, addmm_137, add_178, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        INPUT_SHAPE,
        INPUT_SHAPE,
        RESIDUAL_SHAPE,
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if value.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape0) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected first view shape: {shape0!r}")
    if _shape_tuple(shape1) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected second view shape: {shape1!r}")

    out_shapes = tuple(_shape_tuple(shape) for shape in (shape2, shape3, shape4, shape5))
    for index, shape in enumerate(out_shapes, start=2):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"unexpected output view shape{index}: {shape!r}")

    return addmm_135, addmm_137, add_178, weight, bias, *out_shapes


def oracle_residual_layernorm_aliases(
    addmm_135: torch.Tensor,
    addmm_137: torch.Tensor,
    add_178: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape2: tuple[int, int],
    shape3: tuple[int, int],
    shape4: tuple[int, int],
    shape5: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the full Repro.forward result and return four aliased views."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_layernorm_aliases.py")

    base = torch.empty_strided(
        RESIDUAL_SHAPE,
        BASE_STRIDE,
        device=addmm_135.device,
        dtype=torch.float16,
    )
    _residual_layernorm_aliases_kernel[(ROWS,)](
        addmm_135,
        addmm_137,
        add_178,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=8,
        num_stages=3,
    )
    return (
        base.view(shape2),
        base.view(shape3),
        base.view(shape4),
        base.view(shape5),
    )


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the complete residual-add LayerNorm oracle."""
    return oracle_residual_layernorm_aliases(*_validate_inputs(inputs))


def _check_layout_and_alias(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.dtype == expected_tensor.dtype
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()})"
        )
        ok = ok and layout_ok

    expected_storage_ptrs = {tensor.untyped_storage().data_ptr() for tensor in expected}
    actual_storage_ptrs = {tensor.untyped_storage().data_ptr() for tensor in actual}
    alias_ok = len(expected_storage_ptrs) == 1 and len(actual_storage_ptrs) == 1
    print(
        f"  output aliasing: {'PASS' if alias_ok else 'FAIL'} "
        f"(expected_same_storage={len(expected_storage_ptrs) == 1} "
        f"oracle_same_storage={len(actual_storage_ptrs) == 1})"
    )
    return ok and alias_ok


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
        ok = _check_layout_and_alias(instance, inputs) and ok
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
