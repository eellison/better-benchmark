"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Longformer sliding-window head-layout materialization by copying the overlapped 512-token windows from `[4096, 768]` directly into the final contiguous `[180, 64, 512]` tensor with a shape-specialized 2D Triton layout copy, whereas tuned Inductor already lowers the same view/permute/as_strided/unsqueeze/permute/clone/view scope into one comparable 2D layout-clone kernel; Inductor cannot materially improve this isolated repro today because the required output contract is a newly materialized contiguous clone whose cost is dominated by the mandatory fp16 input reads, fp16 output stores, and one graph-captured launch rather than avoidable intermediate traffic; the fix is BANDWIDTH_BOUND: record this as an at-floor layout-copy case unless broader memory-traffic or launch-overhead work moves both implementations."""
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
    get_shape_key,
    has_stochastic_ops,
)


SEQ_LEN = 4096
HEADS = 12
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
WINDOWS = 15
WINDOW_SIZE = 512
WINDOW_STRIDE = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_Y": 128, "BLOCK_X": 64}, num_warps=8, num_stages=2),
        ],
        key=["Y_SIZE", "WINDOW_SIZE", "HEAD_DIM"],
    )
    @triton.jit
    def _sliding_window_layout_kernel(
        input_ptr,
        output_ptr,
        Y_SIZE: tl.constexpr,
        TOTAL_WINDOWS: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        WINDOW_STRIDE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_Y: tl.constexpr,
        BLOCK_X: tl.constexpr,
    ):
        x_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[None, :]
        y_offsets = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]

        d_offsets = y_offsets % HEAD_DIM
        window = (y_offsets // HEAD_DIM) % TOTAL_WINDOWS
        head = y_offsets // (HEAD_DIM * TOTAL_WINDOWS)

        input_offsets = (
            d_offsets
            + head * HEAD_DIM
            + x_offsets * HIDDEN
            + window * WINDOW_STRIDE * HIDDEN
        )
        mask = (x_offsets < WINDOW_SIZE) & (y_offsets < Y_SIZE)
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + x_offsets + WINDOW_SIZE * y_offsets, values, mask=mask)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_67, shape0, shape1, shape2, shape3, shape4 = inputs
    if not isinstance(addmm_67, torch.Tensor):
        raise TypeError(f"expected addmm_67 tensor, got {type(addmm_67)!r}")
    if tuple(addmm_67.shape) != (SEQ_LEN, HIDDEN):
        raise ValueError(f"unexpected addmm_67 shape: {tuple(addmm_67.shape)}")
    if addmm_67.dtype != torch.float16:
        raise TypeError(f"expected fp16 addmm_67, got {addmm_67.dtype}")
    if addmm_67.stride() != (HIDDEN, 1):
        raise ValueError(f"addmm_67 must be contiguous, got stride={addmm_67.stride()}")

    shape0_tuple = _shape_tuple(shape0, "_shape_param_0")
    shape1_tuple = _shape_tuple(shape1, "_shape_param_1")
    shape2_tuple = _shape_tuple(shape2, "_shape_param_2")
    shape3_tuple = _shape_tuple(shape3, "_shape_param_3")
    shape4_tuple = _shape_tuple(shape4, "_shape_param_4")

    expected_shape0 = (SEQ_LEN, 1, HIDDEN)
    expected_shape1 = (SEQ_LEN, 1, HEADS, HEAD_DIM)
    expected_shape2 = (HEADS, SEQ_LEN, HEAD_DIM)
    expected_shape3 = (HEADS, 8, WINDOW_SIZE, HEAD_DIM)
    expected_shape4 = (HEADS * WINDOWS, HEAD_DIM, WINDOW_SIZE)
    if shape0_tuple != expected_shape0:
        raise ValueError(f"_shape_param_0 {shape0_tuple} != {expected_shape0}")
    if shape1_tuple != expected_shape1:
        raise ValueError(f"_shape_param_1 {shape1_tuple} != {expected_shape1}")
    if shape2_tuple != expected_shape2:
        raise ValueError(f"_shape_param_2 {shape2_tuple} != {expected_shape2}")
    if shape3_tuple != expected_shape3:
        raise ValueError(f"_shape_param_3 {shape3_tuple} != {expected_shape3}")
    if shape4_tuple != expected_shape4:
        raise ValueError(f"_shape_param_4 {shape4_tuple} != {expected_shape4}")

    return (
        addmm_67,
        shape0_tuple,
        shape1_tuple,
        shape2_tuple,
        shape3_tuple,
        shape4_tuple,
    )


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm_67, shape0, shape1, shape2, shape3, shape4 = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm_67, shape0)
    view_default_1 = torch.ops.aten.view.default(view_default, shape1)
    permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0, 2, 3])
    permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3])
    view_default_2 = torch.ops.aten.view.default(permute_default_1, shape2)
    view_default_3 = torch.ops.aten.view.default(view_default_2, shape3)
    as_strided_default = torch.ops.aten.as_strided.default(
        view_default_3,
        [HEADS, WINDOWS, WINDOW_SIZE, HEAD_DIM],
        [HEAD_DIM, WINDOW_STRIDE * HIDDEN, HIDDEN, 1],
    )
    unsqueeze_default = torch.ops.aten.unsqueeze.default(as_strided_default, 4)
    permute_default_2 = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 2, 3])
    permute_default_3 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 4, 3, 2])
    clone_default = torch.ops.aten.clone.default(
        permute_default_3,
        memory_format=torch.contiguous_format,
    )
    return torch.ops.aten.view.default(clone_default, shape4)


@oracle_impl(hardware="H100", shapes="(T([4096, 768], f16), S([4096, 1, 768]), S([4096, 1, 12, 64]), S([12, 4096, 64]), S([12, 8, 512, 64]), S([180, 64, 512]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm_67, _shape0, _shape1, _shape2, _shape3, shape4 = _validate_inputs(inputs)
    if triton is None or not addmm_67.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        shape4,
        (shape4[1] * shape4[2], shape4[2], 1),
        device=addmm_67.device,
        dtype=addmm_67.dtype,
    )
    grid = lambda meta: (
        triton.cdiv(WINDOW_SIZE, meta["BLOCK_X"]),
        triton.cdiv(HEADS * WINDOWS * HEAD_DIM, meta["BLOCK_Y"]),
    )
    _sliding_window_layout_kernel[grid](
        addmm_67,
        output,
        Y_SIZE=HEADS * WINDOWS * HEAD_DIM,
        TOTAL_WINDOWS=WINDOWS,
        WINDOW_SIZE=WINDOW_SIZE,
        WINDOW_STRIDE=WINDOW_STRIDE,
        HEAD_DIM=HEAD_DIM,
        HIDDEN=HIDDEN,
    )
    return output


# --- CLI entry point ---
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


if __name__ == "__main__":
    main()
