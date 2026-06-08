"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle materializes the full `permute(0,2,1,3).clone()._unsafe_view(...).view([B*S,H*D])` contract by writing the transformer head/sequence transpose directly into the final contiguous `[B,S,H*D]` backing storage and returning its `[B*S,H*D]` view, whereas Inductor lowers the decomposed permute-clone-view graph through a generic layout-copy schedule that reaches the same measured floor; Inductor cannot materially improve this repro today because the full-scope work is dominated by the mandatory 538 MB read plus 538 MB write layout copy and launch cost, not by removable arithmetic or scheduler overhead; the fix is BANDWIDTH_BOUND: record this as at floor unless broader layout-copy bandwidth or launch-overhead improvements move the baseline."""
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
from oracle_harness import (  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    get_shape_key,
    has_stochastic_ops,
)


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
            triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8, "BLOCK_COLS": 256}, num_warps=8, num_stages=3),
        ],
        key=["SEQ", "HEADS", "HEAD_DIM"],
    )
    @triton.jit
    def oracle_kernel(
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
        seq = rows - batch * SEQ
        head = cols // HEAD_DIM
        dim = cols - head * HEAD_DIM

        input_offsets = (
            batch * SEQ * HIDDEN
            + head * SEQ * HEAD_DIM
            + seq * HEAD_DIM
            + dim
        )
        output_offsets = rows * HIDDEN + cols

        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(x)!r}")
    if x.device.type != "cuda":
        raise RuntimeError("oracle_head_flatten_layout.py expects CUDA tensor inputs")
    if x.dtype != torch.float32:
        raise TypeError(f"unexpected input dtype {x.dtype}, expected torch.float32")
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 input, got shape={tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(x.stride())}")

    batch, heads, seq, dim = (int(v) for v in x.shape)
    output_shape = _shape_tuple(shape_param)
    expected_shape = (batch * seq, heads * dim)
    if output_shape != expected_shape:
        raise ValueError(
            f"unexpected output shape parameter {output_shape}, expected {expected_shape}"
        )

    return x, batch, heads, seq, dim, output_shape


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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_head_flatten_layout.py")

    x, batch, heads, seq, dim, output_shape = _validate_inputs(inputs)
    hidden = heads * dim
    backing = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=x.device,
        dtype=x.dtype,
    )
    output = backing.view(output_shape)

    rows = batch * seq
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_ROWS"]),
        triton.cdiv(hidden, meta["BLOCK_COLS"]),
    )
    oracle_kernel[grid](
        x,
        output,
        ROWS=rows,
        SEQ=seq,
        HEADS=heads,
        HEAD_DIM=dim,
        HIDDEN=hidden,
    )
    return output


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_layout_and_alias(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    all_pass = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and tuple(actual_tensor.stride()) == tuple(expected_tensor.stride())
            and actual_tensor.storage_offset() == expected_tensor.storage_offset()
            and actual_tensor.dtype == expected_tensor.dtype
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={tuple(actual_tensor.stride())} "
            f"storage_offset={actual_tensor.storage_offset()} dtype={actual_tensor.dtype})"
        )
        all_pass = all_pass and layout_ok

        expected_base = expected_tensor._base
        actual_base = actual_tensor._base
        base_ok = (
            expected_base is not None
            and actual_base is not None
            and tuple(actual_base.shape) == tuple(expected_base.shape)
            and tuple(actual_base.stride()) == tuple(expected_base.stride())
            and actual_base.storage_offset() == expected_base.storage_offset()
            and actual_tensor.data_ptr() == actual_base.data_ptr()
        )
        print(
            f"  output {index} view base: {'PASS' if base_ok else 'FAIL'} "
            f"(base_shape={None if actual_base is None else list(actual_base.shape)})"
        )
        all_pass = all_pass and base_ok

    return all_pass


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
        ok = _check_layout_and_alias(instance, inputs) and ok
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
