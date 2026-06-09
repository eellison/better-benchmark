"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Qwen/Llama permute -> contiguous clone -> unsafe_view -> final view Repro.forward contract by directly materializing the contiguous bf16[2048, 2048] output from the captured bf16[B, H, 512, D] input with a shape-specialized Triton head-layout copy, whereas Inductor currently lowers the decomposed view/permute/clone/view chain as a generic pointwise layout materialization with scalar index decoding; Inductor cannot do this today because its pointwise scheduler/codegen does not recognize the static attention-head flattening idiom permute(0, 2, 1, 3).clone().view(B*S, H*D) and emit head-vectorized contiguous loads into the final row-major storage; the fix is NEW_PATTERN: add a specialized layout-copy lowering for permute-contiguous-flatten materializations whose hidden dimension is preserved while heads and sequence are swapped."""
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
            triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 512}, num_warps=8, num_stages=3),
        ],
        key=["HEADS", "HEAD_DIM"],
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
        seq_idx = rows - batch * SEQ
        head = cols // HEAD_DIM
        dim = cols - head * HEAD_DIM

        input_offsets = (
            batch * (SEQ * HIDDEN)
            + head * (SEQ * HEAD_DIM)
            + seq_idx * HEAD_DIM
            + dim
        )
        output_offsets = rows * HIDDEN + cols
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    arg323_1, shape_param = inputs
    if not isinstance(arg323_1, torch.Tensor):
        raise TypeError(f"expected arg323_1 tensor, got {type(arg323_1)!r}")
    if arg323_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if arg323_1.dtype != torch.bfloat16:
        raise TypeError(f"expected bf16 input, got {arg323_1.dtype}")
    if arg323_1.dim() != 4:
        raise ValueError(f"expected rank-4 input, got shape={tuple(arg323_1.shape)}")
    if not arg323_1.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(arg323_1.stride())}")

    batch, heads, seq, head_dim = (int(dim) for dim in arg323_1.shape)
    rows = batch * seq
    hidden = heads * head_dim
    out_shape = _shape_tuple(shape_param, "_shape_param_0")
    expected_shape = (rows, hidden)
    if out_shape != expected_shape:
        raise ValueError(f"_shape_param_0 {out_shape} != expected {expected_shape}")
    if out_shape != (2048, 2048):
        raise ValueError(f"unexpected output shape {out_shape}; expected (2048, 2048)")
    if seq != 512:
        raise ValueError(f"unexpected sequence length {seq}; expected 512")
    if (heads, head_dim) not in ((16, 128), (32, 64)):
        raise ValueError(
            f"unexpected head layout heads={heads} head_dim={head_dim}; "
            "expected (16, 128) or (32, 64)"
        )

    return arg323_1, out_shape, rows, seq, heads, head_dim, hidden


@oracle_impl(hardware="H100", shapes="(T([4, 16, 512, 128], bf16), S([2048, 2048]))")
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

    arg323_1, out_shape, rows, seq, heads, head_dim, hidden = _validate_inputs(inputs)
    output = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg323_1.device,
        dtype=arg323_1.dtype,
    )
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_ROWS"]),
        triton.cdiv(hidden, meta["BLOCK_COLS"]),
    )
    oracle_kernel[grid](
        arg323_1,
        output,
        ROWS=rows,
        SEQ=seq,
        HEADS=heads,
        HEAD_DIM=head_dim,
        HIDDEN=hidden,
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
