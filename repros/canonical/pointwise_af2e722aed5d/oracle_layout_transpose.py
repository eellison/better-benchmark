"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete attention-head layout materialization returned by Repro.forward, resolving the input view shape and writing the `view(B,H,S,D).permute(0,2,1,3).clone().view(B*S,H*D)` result directly as a fresh contiguous tensor with simplified affine indexing, whereas Inductor lowers the same clone/view contract to a generic layout-copy kernel in the same measured memory-traffic envelope; Inductor cannot materially do less work today because the isolated repro must return a non-aliasing contiguous clone, so the mandatory input read and output write dominate; the fix is BANDWIDTH_BOUND: treat this as at floor unless broader layout-copy bandwidth, indexing-code polish, or launch-overhead improvements move both implementations."""
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

from oracle_harness import (
    oracle_impl,  # noqa: E402
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


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 512}, num_warps=8, num_stages=3),
        ],
        key=["SEQ", "HEADS", "HEAD_DIM"],
    )
    @triton.jit
    def _head_sequence_layout_kernel(
        input_ptr,
        output_ptr,
        ROWS: tl.constexpr,
        SEQ: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
        mask = (rows < ROWS) & (cols < HIDDEN)

        batch = rows // SEQ
        seq_idx = rows - batch * SEQ
        head = cols // HEAD_DIM
        dim = cols - head * HEAD_DIM

        input_offsets = (
            batch * (HEADS * SEQ * HEAD_DIM)
            + head * (SEQ * HEAD_DIM)
            + seq_idx * HEAD_DIM
            + dim
        )
        output_offsets = rows * HIDDEN + cols

        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(value: Any, numel: int, name: str) -> tuple[int, ...]:
    dims = list(_shape_tuple(value, name))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"{name} has more than one inferred dimension: {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known <= 0 or numel % known != 0:
            raise ValueError(f"cannot infer {name}={dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known

    product = 1
    for dim in dims:
        product *= dim
    if product != numel:
        raise ValueError(f"{name}={tuple(dims)} has {product} elements, expected {numel}")
    return tuple(dims)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int], int, int, int, int]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    bmm_23, shape0, shape1, shape2 = inputs
    if not isinstance(bmm_23, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if bmm_23.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if bmm_23.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects torch.float32 input, got {bmm_23.dtype}")
    if bmm_23.ndim != 3:
        raise ValueError(f"{REPRO_ID} expects rank-3 input, got shape={tuple(bmm_23.shape)}")
    if not bmm_23.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(bmm_23.stride())}")
    if bmm_23.storage_offset() != 0:
        raise ValueError(f"{REPRO_ID} expects storage_offset=0, got {bmm_23.storage_offset()}")

    numel = int(bmm_23.numel())
    batch, heads, seq, head_dim = _resolve_view_shape(shape0, numel, "_shape_param_0")
    if tuple(bmm_23.shape) != (batch * heads, seq, head_dim):
        raise ValueError(
            f"_shape_param_0={(batch, heads, seq, head_dim)} is incompatible "
            f"with input shape={tuple(bmm_23.shape)}"
        )

    hidden = heads * head_dim
    middle_shape = _resolve_view_shape(shape1, numel, "_shape_param_1")
    expected_middle = (batch, seq, hidden)
    if middle_shape != expected_middle:
        raise ValueError(f"_shape_param_1={middle_shape} != expected {expected_middle}")

    output_shape = _resolve_view_shape(shape2, numel, "_shape_param_2")
    expected_output = (batch * seq, hidden)
    if output_shape != expected_output:
        raise ValueError(f"_shape_param_2={output_shape} != expected {expected_output}")

    return bmm_23, output_shape, (hidden, 1), batch * seq, seq, heads, head_dim


@oracle_impl(hardware="H100", shapes="(T([512, 512, 64], f32), S([8, 64, 512, 64]), S([8, 512, -1]), S([4096, 4096]))")
def oracle_forward(inputs):
    """Run the full Repro.forward layout materialization scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    bmm_23, output_shape, output_stride, rows, seq, heads, head_dim = _validate_inputs(inputs)
    hidden = heads * head_dim
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=bmm_23.device,
        dtype=bmm_23.dtype,
    )

    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_ROWS"]),
        triton.cdiv(hidden, meta["BLOCK_COLS"]),
    )
    _head_sequence_layout_kernel[grid](
        bmm_23,
        output,
        ROWS=rows,
        SEQ=seq,
        HEADS=heads,
        HEAD_DIM=head_dim,
        HIDDEN=hidden,
    )
    return output


def _check_layout_and_exact_values(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(actual.shape) == tuple(eager.shape)
        and tuple(actual.stride()) == tuple(eager.stride())
        and actual.dtype == eager.dtype
        and actual.storage_offset() == eager.storage_offset()
    )
    values_ok = torch.equal(eager, actual)
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={list(actual.stride())} "
        f"dtype={actual.dtype} storage_offset={actual.storage_offset()})"
    )
    print(f"  output 0 exact: {'PASS' if values_ok else 'FAIL'}")
    return layout_ok and values_ok


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
        ok = _check_layout_and_exact_values(instance, inputs) and ok
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
