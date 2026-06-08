"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT QKV layout materialization returned by Repro.forward, including the `[128,12,64,128] -> [128,128,12,64] -> [16384,768]` contiguous clone backing storage, the returned `[768,16384]` permute view with eager-compatible stride, and the sibling `[768]` f32 column sum with a Triton copy-and-batch-partial pass plus a small finalizer, while Inductor already emits the same essential two-pass fused layout-plus-partial-sum schedule; Inductor cannot materially improve this local scope today because the remaining work is dominated by the mandatory f32 layout copy and f32 reductions over the same input values, so the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader transpose/layout-copy bandwidth work moves both implementations."""
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


# --- Oracle kernel(s) ---

COPY_XBLOCK = 32
COPY_SBLOCK = 128
FINAL_B_BLOCK = 128
FINAL_C_BLOCK = 16


if triton is not None:

    @triton.jit
    def _copy_and_batch_sum_kernel(
        input_ptr,
        base_ptr,
        partial_ptr,
        XNUMEL: tl.constexpr,
        C: tl.constexpr,
        S: tl.constexpr,
        XBLOCK: tl.constexpr,
        SBLOCK: tl.constexpr,
    ):
        x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        seq = tl.arange(0, SBLOCK)
        mask = (x[:, None] < XNUMEL) & (seq[None, :] < S)

        batch = x // C
        col = x - batch * C
        values = tl.load(input_ptr + x[:, None] * S + seq[None, :], mask=mask, other=0.0).to(tl.float32)
        base_offsets = batch[:, None] * S * C + seq[None, :] * C + col[:, None]
        tl.store(base_ptr + base_offsets, values, mask=mask)

        partial = tl.sum(tl.where(mask, values, 0.0), axis=1)
        tl.store(partial_ptr + x, partial, mask=x < XNUMEL)

    @triton.jit
    def _finalize_sum_kernel(
        partial_ptr,
        out_sum_ptr,
        B: tl.constexpr,
        C: tl.constexpr,
        B_BLOCK: tl.constexpr,
        C_BLOCK: tl.constexpr,
    ):
        cols = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
        batches = tl.arange(0, B_BLOCK)
        mask = (batches[:, None] < B) & (cols[None, :] < C)
        offsets = batches[:, None] * C + cols[None, :]
        values = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        total = tl.sum(values, axis=0)
        tl.store(out_sum_ptr + cols, total, mask=cols < C)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known
    resolved = tuple(dims)
    if _numel(resolved) != numel:
        raise ValueError(f"shape {resolved} has {_numel(resolved)} elements, expected {numel}")
    return resolved


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, int, int, int, int, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    bmm_46, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(bmm_46, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if bmm_46.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {bmm_46.dtype}")
    if bmm_46.ndim != 3:
        raise ValueError(f"{REPRO_ID} expects rank-3 input, got shape={tuple(bmm_46.shape)}")
    if not bmm_46.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(bmm_46.stride())}")

    b, h, d, s = _resolve_view_shape(shape0, int(bmm_46.numel()))
    expected_input_shape = (b * h, d, s)
    if tuple(bmm_46.shape) != expected_input_shape:
        raise ValueError(f"input shape {tuple(bmm_46.shape)} does not match {expected_input_shape}")

    c = h * d
    m = b * s
    if _resolve_view_shape(shape1, int(bmm_46.numel())) != (b, s, c):
        raise ValueError(f"unexpected second view shape {shape1}")
    if _resolve_view_shape(shape2, int(bmm_46.numel())) != (m, c):
        raise ValueError(f"unexpected final view shape {shape2}")
    if _resolve_view_shape(shape3, c) != (c,):
        raise ValueError(f"unexpected sum view shape {shape3}")

    return bmm_46, m, c, s, h, d, b


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]):
    bmm_46, shape0, shape1, shape2, shape3 = inputs
    view_default = torch.ops.aten.view.default(bmm_46, shape0)
    permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2])
    permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3])
    view_default_1 = torch.ops.aten.view.default(permute_default_1, shape1)
    clone_default = torch.ops.aten.clone.default(view_default_1, memory_format=torch.contiguous_format)
    view_default_2 = torch.ops.aten.view.default(clone_default, shape2)
    permute_default_2 = torch.ops.aten.permute.default(view_default_2, [1, 0])
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True)
    view_default_3 = torch.ops.aten.view.default(sum_dim_int_list, shape3)
    return (permute_default_2, view_default_3)


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
    bmm_46, m, c, s, _h, _d, b = _validate_inputs(inputs)
    if triton is None or not bmm_46.is_cuda:
        return _torch_full_scope(inputs)

    base = torch.empty((m, c), device=bmm_46.device, dtype=torch.float32)
    out_sum = torch.empty((c,), device=bmm_46.device, dtype=torch.float32)
    xnumel = b * c
    partial = torch.empty((b, c), device=bmm_46.device, dtype=torch.float32)

    _copy_and_batch_sum_kernel[(triton.cdiv(xnumel, COPY_XBLOCK),)](
        bmm_46,
        base,
        partial,
        XNUMEL=xnumel,
        C=c,
        S=s,
        XBLOCK=COPY_XBLOCK,
        SBLOCK=COPY_SBLOCK,
        num_warps=8,
    )
    _finalize_sum_kernel[(triton.cdiv(c, FINAL_C_BLOCK),)](
        partial,
        out_sum,
        B=b,
        C=c,
        B_BLOCK=FINAL_B_BLOCK,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=8,
    )
    return (base.permute(1, 0), out_sum)


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
