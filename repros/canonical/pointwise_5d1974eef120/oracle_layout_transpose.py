"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete BART attention layout scope by directly materializing the required contiguous float32[2048, 768] output for the permute -> contiguous clone -> unsafe_view -> final view chain in one row-specialized Triton copy, whereas Inductor currently emits a single generic clone/permute pointwise kernel with heavier logical index decoding for the same required materialization; Inductor cannot do this today because its layout algebra does not collapse this fixed permute/clone/view chain into an output-row-contiguous transpose-copy template with shape-specialized affine indexing; the fix is ALGEBRAIC_ELIMINATION: canonicalize such layout-only chains to a direct dense materialization kernel while preserving the exact returned shape and stride."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


BATCH = 4
HEADS = 12
SEQ = 512
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
ROWS = BATCH * SEQ
ROW_BLOCK_N = 1024
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(x)!r}")
    if tuple(x.shape) != (BATCH, HEADS, SEQ, HEAD_DIM):
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != (HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1):
        raise ValueError(f"unexpected input stride: {tuple(x.stride())}")
    if x.storage_offset() != 0:
        raise ValueError(f"unexpected input storage_offset: {x.storage_offset()}")
    if x.dtype is not torch.float32:
        raise TypeError(f"expected torch.float32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("oracle_layout_transpose.py expects CUDA inputs")

    out_shape = _shape_tuple(shape_param, "_shape_param_0")
    if out_shape != (ROWS, HIDDEN):
        raise ValueError(f"unexpected _shape_param_0: {out_shape}")
    return x, out_shape


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _layout_transpose_kernel(
        input_ptr,
        output_ptr,
        TOTAL_ROWS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < TOTAL_ROWS) & (cols[None, :] < 768)

        batch = rows // 512
        seq = rows - batch * 512
        head = cols // 64
        dim = cols - head * 64

        input_offsets = (
            batch[:, None] * 393216
            + head[None, :] * 32768
            + seq[:, None] * 64
            + dim[None, :]
        )
        output_offsets = rows[:, None] * 768 + cols[None, :]

        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([4, 12, 512, 64], f32), S([2048, 768]))")
def oracle_forward(inputs):
    """Run the full permute -> contiguous clone -> unsafe_view -> view scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, out_shape = _validate_inputs(tuple(inputs))
    output = torch.empty_strided(
        out_shape,
        (HIDDEN, 1),
        device=x.device,
        dtype=x.dtype,
    )
    _layout_transpose_kernel[(ROWS,)](
        x,
        output,
        TOTAL_ROWS=ROWS,
        BLOCK_N=ROW_BLOCK_N,
        BLOCK_ROWS=1,
        num_warps=4,
        num_stages=1,
    )
    return output


def _as_tensor_tuple(value):
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, torch.Tensor))
    raise TypeError(f"expected tensor output, got {type(value)!r}")


def _check_output_layout(instance, inputs):
    with torch.no_grad():
        eager_outputs = _as_tensor_tuple(instance(*inputs))
        oracle_outputs = _as_tensor_tuple(oracle_forward(inputs))
        if any(output.is_cuda for output in oracle_outputs):
            torch.cuda.synchronize()

    if len(eager_outputs) != len(oracle_outputs):
        print(
            f"  layout: FAIL output count oracle={len(oracle_outputs)} "
            f"eager={len(eager_outputs)}"
        )
        return False

    ok = True
    for index, (eager, oracle) in enumerate(zip(eager_outputs, oracle_outputs)):
        layout_ok = (
            tuple(oracle.shape) == tuple(eager.shape)
            and oracle.dtype == eager.dtype
            and tuple(oracle.stride()) == tuple(eager.stride())
            and oracle.storage_offset() == eager.storage_offset()
        )
        status = "PASS" if layout_ok else "FAIL"
        print(
            f"  output {index} layout: {status} "
            f"(shape={list(oracle.shape)} dtype={oracle.dtype} "
            f"stride={tuple(oracle.stride())} storage_offset={oracle.storage_offset()})"
        )
        ok = ok and layout_ok
    return ok


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
        ok = _check_output_layout(instance, inputs) and ok
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
