"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full GPT-2-large `permute.clone.view.permute` scope by materializing the required contiguous `[B, S, H, D]` clone buffer with 64-wide Triton vector copies and returning the same `[H*D, B*S]` transpose view, whereas Inductor already emits a full-scope fused clone-permute copy and only carries extra generic indexing overhead; Inductor cannot remove the dominant work today because the isolated repro contract returns a view backed by a newly materialized contiguous clone buffer, so the input read and clone write traffic are required; the fix is BANDWIDTH_BOUND: record this as a layout-copy floor and only pursue minor index-specialization or clone-elision improvements when consumers outside this repro make the materialization unnecessary."""
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
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_VECTORS"],
    )
    @triton.jit
    def _layout_transpose_clone_kernel(
        input_ptr,
        clone_ptr,
        TOTAL_VECTORS: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        vector = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        dim = tl.arange(0, D)
        mask = vector < TOTAL_VECTORS

        head = vector % H
        tmp = vector // H
        seq = tmp % S
        batch = tmp // S

        input_offsets = (
            batch[:, None] * H * S * D
            + head[:, None] * S * D
            + seq[:, None] * D
            + dim[None, :]
        )
        clone_offsets = vector[:, None] * D + dim[None, :]
        values = tl.load(input_ptr + input_offsets, mask=mask[:, None], other=0.0)
        tl.store(clone_ptr + clone_offsets, values, mask=mask[:, None])


def _expect_shape_param(shape_param, hidden: int) -> None:
    if list(shape_param) != [-1, hidden]:
        raise ValueError(f"unexpected shape parameter {shape_param}, expected [-1, {hidden}]")


def _check_output_layout(output: torch.Tensor, batch: int, seq: int, hidden: int) -> bool:
    return (
        tuple(output.shape) == (hidden, batch * seq)
        and tuple(output.stride()) == (1, hidden)
        and output.storage_offset() == 0
    )


@oracle_impl(hardware="H100", shapes="(T([4, 20, 512, 64], f32), S([-1, 1280]))")
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
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 input, got shape={tuple(x.shape)}")
    if x.dtype is not torch.float32:
        raise ValueError(f"expected float32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA tensor inputs")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(x.stride())}")

    batch, heads, seq, dim = (int(v) for v in x.shape)
    hidden = heads * dim
    _expect_shape_param(shape_param, hidden)
    if dim != 64:
        raise ValueError(f"expected captured head dim 64, got {dim}")

    clone = torch.empty_strided(
        (batch, seq, heads, dim),
        (seq * hidden, hidden, dim, 1),
        device=x.device,
        dtype=x.dtype,
    )
    total_vectors = batch * seq * heads
    grid = lambda meta: (triton.cdiv(total_vectors, meta["BLOCK_M"]),)
    _layout_transpose_clone_kernel[grid](
        x,
        clone,
        TOTAL_VECTORS=total_vectors,
        S=seq,
        H=heads,
        D=dim,
    )
    return clone.view(batch * seq, hidden).t()


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
        x = inputs[0]
        with torch.no_grad():
            layout_output = oracle_forward(inputs)
            if layout_output.is_cuda:
                torch.cuda.synchronize()
        layout_ok = _check_output_layout(
            layout_output,
            int(x.shape[0]),
            int(x.shape[2]),
            int(x.shape[1]) * int(x.shape[3]),
        )
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={list(layout_output.stride())})"
        )
        ok = ok and layout_ok
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
