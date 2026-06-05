"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full bf16 grouped-KV materialization returned by Repro.forward, including the view/permute/unsqueeze/expand/clone/view layout contract, with a shape-specialized Triton copy that loads each source KV-head tile once and stores it into every expanded group slot of the contiguous `[B, H, S, D]` output, whereas Inductor lowers the same zero-stride expand clone as a generic layout-conversion copy rather than a grouped multi-store materialization. Inductor cannot do this today because its scheduler/codegen has no guarded grouped-KV expand-clone template that reuses one source load across the expanded group stores while preserving the exact contiguous output layout; the fix is NEW_PATTERN: add a benchmark-gated grouped-KV materialization lowering for zero-stride expand+clone layouts, falling back to the generic copy when the bandwidth win disappears."""
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


MAX_GROUPS = 8


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
        ],
        key=["S", "D", "KV", "GROUPS"],
    )
    @triton.jit
    def _grouped_kv_materialize_kernel(
        src_ptr,
        out_ptr,
        S: tl.constexpr,
        D: tl.constexpr,
        KV: tl.constexpr,
        GROUPS: tl.constexpr,
        KVD: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b_kv = tl.program_id(0)
        tile = tl.program_id(1)
        batch = b_kv // KV
        kv_head = b_kv - batch * KV

        offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < S * D
        seq = offsets // D
        dim = offsets - seq * D

        values = tl.load(
            src_ptr + (batch * S + seq) * KVD + kv_head * D + dim,
            mask=mask,
            other=0.0,
        )

        out_head = kv_head * GROUPS
        out_base = (batch * (KV * GROUPS) + out_head) * S * D + offsets
        for group in tl.static_range(0, 8):
            if group < GROUPS:
                tl.store(out_ptr + out_base + group * S * D, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs):
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    src, view_shape, kv_view_shape, expand_shape, out_shape = inputs
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"mm_191 must be a tensor, got {type(src)!r}")
    if not src.is_cuda:
        raise RuntimeError("mm_191 must be a CUDA tensor for this Triton oracle")
    if src.dtype != torch.bfloat16:
        raise TypeError(f"mm_191 has dtype {src.dtype}, expected torch.bfloat16")
    if len(src.shape) != 2:
        raise ValueError(f"mm_191 must be rank 2, got shape {tuple(src.shape)}")
    if tuple(src.stride()) != (src.shape[1], 1):
        raise ValueError(
            f"mm_191 has stride {tuple(src.stride())}, expected {(src.shape[1], 1)}"
        )

    view_shape = _shape_tuple(view_shape)
    kv_view_shape = _shape_tuple(kv_view_shape)
    expand_shape = _shape_tuple(expand_shape)
    out_shape = _shape_tuple(out_shape)

    if len(view_shape) != 3 or len(kv_view_shape) != 4:
        raise ValueError(
            f"unexpected view shape params: {view_shape!r}, {kv_view_shape!r}"
        )
    if len(expand_shape) != 5 or len(out_shape) != 4:
        raise ValueError(
            f"unexpected output shape params: {expand_shape!r}, {out_shape!r}"
        )

    batch, kv_heads, groups, seq, head_dim = expand_shape
    if kv_view_shape != (batch, seq, -1, head_dim):
        raise ValueError(
            f"kv view shape mismatch: expected {(batch, seq, -1, head_dim)}, "
            f"got {kv_view_shape}"
        )
    if out_shape != (batch, kv_heads * groups, seq, head_dim):
        raise ValueError(
            f"output view shape mismatch: expected "
            f"{(batch, kv_heads * groups, seq, head_dim)}, got {out_shape}"
        )
    if view_shape != (batch, seq, kv_heads * head_dim):
        raise ValueError(
            f"first view shape mismatch: expected "
            f"{(batch, seq, kv_heads * head_dim)}, got {view_shape}"
        )
    if tuple(src.shape) != (batch * seq, kv_heads * head_dim):
        raise ValueError(
            f"mm_191 shape mismatch: expected "
            f"{(batch * seq, kv_heads * head_dim)}, got {tuple(src.shape)}"
        )
    if groups < 1 or groups > MAX_GROUPS:
        raise ValueError(f"unsupported group count {groups}; max is {MAX_GROUPS}")

    return src, batch, seq, kv_heads, groups, head_dim, out_shape


def oracle_forward(inputs):
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_grouped_kv_materialize.py")

    src, batch, seq, kv_heads, groups, head_dim, out_shape = _validate_inputs(inputs)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * seq * head_dim, seq * head_dim, head_dim, 1),
        device=src.device,
        dtype=src.dtype,
    )

    grid = lambda meta: (batch * kv_heads, triton.cdiv(seq * head_dim, meta["BLOCK_N"]))
    _grouped_kv_materialize_kernel[grid](
        src,
        out,
        S=seq,
        D=head_dim,
        KV=kv_heads,
        GROUPS=groups,
        KVD=kv_heads * head_dim,
    )
    return out


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
