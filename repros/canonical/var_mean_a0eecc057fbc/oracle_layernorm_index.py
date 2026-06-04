"""
Oracle for var_mean_a0eecc057fbc

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the
full Repro.forward contract by reducing all 64 rows of
`add_91 + view(addmm_47)` to produce the returned `rsqrt(var + eps) / 768`
tensor, while applying the affine LayerNorm epilogue only to the final sequence
row that survives `aten.index.Tensor(..., -1)`, whereas Inductor lowers the
decomposed LayerNorm/index graph as a full row-normalization producer before the
advanced-index consumer; Inductor cannot do this today because its scheduler does
not push constant/single-row advanced-index demand backward through a multi-output
normalization whose sibling `div_tensor` still needs all row statistics; the fix
is ALGEBRAIC_ELIMINATION: add a guarded demand-propagation/index-elimination
rewrite for LayerNorm epilogues that keeps all required stats but restricts
dead normalized-row materialization to the indexed rows.
"""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 64
HIDDEN = 768
EPS = 1.0e-5
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(int(size), 1)
    return tuple(reversed(strides))


if triton is not None:

    @triton.jit
    def _layernorm_index_kernel(
        addmm_ptr,
        add_ptr,
        weight_ptr,
        bias_ptr,
        out_indexed_ptr,
        out_div_ptr,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        EPS_: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        col_mask = cols < HIDDEN_
        linear_offsets = row * HIDDEN_ + cols

        values = (
            tl.load(addmm_ptr + linear_offsets, mask=col_mask, other=0.0).to(tl.float32)
            + tl.load(add_ptr + linear_offsets, mask=col_mask, other=0.0).to(tl.float32)
        )
        mean = tl.sum(values, axis=0) / HIDDEN_
        centered = values - mean
        var = tl.sum(centered * centered, axis=0) / HIDDEN_
        invstd = tl.rsqrt(var + EPS_)

        tl.store(out_div_ptr + row, invstd * (1.0 / HIDDEN_))

        is_indexed_row = row == (ROWS_ - 1)
        weight = tl.load(weight_ptr + cols, mask=col_mask & is_indexed_row, other=0.0).to(
            tl.float32
        )
        bias = tl.load(bias_ptr + cols, mask=col_mask & is_indexed_row, other=0.0).to(
            tl.float32
        )
        out = centered * invstd * weight + bias
        tl.store(out_indexed_ptr + cols, out, mask=col_mask & is_indexed_row)


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 6:
        raise ValueError(f"expected 6 inputs, got {len(inputs)}")

    addmm_47, add_91, arg147_1, arg148_1, shape0, shape1 = inputs
    tensor_inputs = (addmm_47, add_91, arg147_1, arg148_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first four repro inputs must be tensors")
    if tuple(addmm_47.shape) != (ROWS, HIDDEN):
        raise ValueError(f"addmm_47 shape {tuple(addmm_47.shape)} != {(ROWS, HIDDEN)}")
    if tuple(add_91.shape) != (1, ROWS, HIDDEN):
        raise ValueError(f"add_91 shape {tuple(add_91.shape)} != {(1, ROWS, HIDDEN)}")
    if tuple(arg147_1.shape) != (HIDDEN,) or tuple(arg148_1.shape) != (HIDDEN,):
        raise ValueError(f"affine input shapes must be {(HIDDEN,)}")
    if any(value.dtype != torch.float32 for value in tensor_inputs):
        raise TypeError("all tensor inputs must be torch.float32")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        raise ValueError("all tensor inputs must be contiguous")

    shape0_tuple = tuple(int(dim) for dim in shape0)
    shape1_tuple = tuple(int(dim) for dim in shape1)
    if shape0_tuple != (1, ROWS, HIDDEN) or shape1_tuple != (1, HIDDEN):
        raise ValueError(f"unexpected shape parameters: {shape0!r}, {shape1!r}")

    return addmm_47, add_91, arg147_1, arg148_1, shape0_tuple, shape1_tuple


def oracle_forward(inputs):
    """Run the full Repro.forward computation with indexed-row elimination.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same two outputs: indexed affine LayerNorm row and all-row div_tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    addmm_47, add_91, arg147_1, arg148_1, _shape0, shape1 = _validate_inputs(inputs)
    out_indexed = torch.empty_strided(
        shape1,
        _contiguous_strides(shape1),
        device=addmm_47.device,
        dtype=addmm_47.dtype,
    )
    out_div = torch.empty_strided(
        (1, ROWS, 1),
        (ROWS, 1, 1),
        device=addmm_47.device,
        dtype=addmm_47.dtype,
    )

    _layernorm_index_kernel[(ROWS,)](
        addmm_47,
        add_91,
        arg147_1,
        arg148_1,
        out_indexed,
        out_div,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        EPS_=EPS,
        BLOCK_H=1024,
        num_warps=8,
    )
    return out_indexed, out_div


def _check_layout(inputs) -> bool:
    with torch.no_grad():
        expected = get_repro_instance()(*inputs)
        actual = oracle_forward(inputs)
    if actual[0].is_cuda:
        torch.cuda.synchronize()

    ok = True
    for idx, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.stride() == expected_tensor.stride()
        )
        print(
            f"  output {idx} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()})"
        )
        ok = ok and layout_ok
    return ok


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
        ok = _check_layout(inputs) and ok
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
