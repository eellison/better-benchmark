"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin QKV split/layout scope returned by Repro.forward, including the V layout clone, scaled-Q layout clone, K-transpose layout clone, and final transpose-view outputs with eager-compatible strides and non-aliasing fresh storages, by grouping Q/V and materializing K in storage order, whereas Inductor's compiled path is already slightly faster on the same mandatory Q/K/V layout-copy traffic; Inductor cannot be assigned a material local gap today because the output contract requires reading all three QKV planes, scaling Q, and writing three fresh backing storages, so the remaining cost is dominated by required memory movement; the fix is BANDWIDTH_BOUND: no narrow scheduler or algebraic change is indicated beyond broader layout-copy bandwidth/codegen improvements."""
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


SCALE = 0.1767766952966369


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
            triton.Config({"YBLOCK": 8, "XBLOCK": 32}, num_warps=1, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 32}, num_warps=1, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 32}, num_warps=2, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 64, "XBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 64}, num_warps=2, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 64}, num_warps=4, num_stages=3),
        ],
        key=["YNUMEL", "TOKENS", "HEADS", "HEAD_DIM"],
    )
    @triton.jit
    def _qv_split_layout_kernel(
        input_ptr,
        v_base_ptr,
        q_base_ptr,
        YNUMEL: tl.constexpr,
        TOKENS: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCALE_VALUE: tl.constexpr,
        YBLOCK: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        yidx = tl.program_id(0) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
        didx = tl.arange(0, XBLOCK)[None, :]
        mask = (yidx < YNUMEL) & (didx < HEAD_DIM)

        token = yidx % TOKENS
        tmp = yidx // TOKENS
        head = tmp % HEADS
        batch = tmp // HEADS

        qkv_plane = HEADS * HEAD_DIM
        input_base = (batch * TOKENS + token) * (3 * qkv_plane) + head * HEAD_DIM + didx
        q_values = tl.load(input_ptr + input_base, mask=mask, other=0.0) * SCALE_VALUE
        v_values = tl.load(input_ptr + input_base + 2 * qkv_plane, mask=mask, other=0.0)

        qv_offsets = yidx * HEAD_DIM + didx

        tl.store(v_base_ptr + qv_offsets, v_values, mask=mask)
        tl.store(q_base_ptr + qv_offsets, q_values, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        ],
        key=["N", "TOKENS", "HEADS", "HEAD_DIM"],
    )
    @triton.jit
    def _k_split_layout_kernel(
        input_ptr,
        k_base_ptr,
        N: tl.constexpr,
        TOKENS: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        token = offsets % TOKENS
        tmp = offsets // TOKENS
        dim = tmp % HEAD_DIM
        tmp = tmp // HEAD_DIM
        head = tmp % HEADS
        batch = tmp // HEADS

        qkv_plane = HEADS * HEAD_DIM
        input_offsets = (batch * TOKENS + token) * (3 * qkv_plane) + qkv_plane + head * HEAD_DIM + dim
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(k_base_ptr + offsets, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
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


def _validate_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int, int, int], int, int, int, int]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects nine inputs, got {len(inputs)}")

    addmm_92, shape0, shape1, shape2, shape3, shape4, shape5, shape6, shape7 = inputs
    if not isinstance(addmm_92, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not addmm_92.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if addmm_92.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {addmm_92.dtype}")
    if addmm_92.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects rank-2 input, got shape={tuple(addmm_92.shape)}")
    if not addmm_92.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(addmm_92.stride())}")

    rows = int(addmm_92.shape[0])
    hidden = int(addmm_92.shape[1])
    numel = int(addmm_92.numel())

    batch, tokens, view_hidden = _resolve_view_shape(shape0, numel)
    if (batch * tokens, view_hidden) != (rows, hidden):
        raise ValueError(
            f"first view shape {(batch, tokens, view_hidden)} is incompatible "
            f"with input shape={tuple(addmm_92.shape)}"
        )

    batch1, tokens1, qkv, heads, head_dim = _resolve_view_shape(shape1, numel)
    if (batch1, tokens1) != (batch, tokens) or qkv != 3 or heads * head_dim * qkv != hidden:
        raise ValueError(
            f"QKV view shape {(batch1, tokens1, qkv, heads, head_dim)} is incompatible "
            f"with first view {(batch, tokens, hidden)}"
        )

    qv_base_shape = (batch, heads, tokens, head_dim)
    k_base_shape = (batch, heads, head_dim, tokens)
    expected_qv_expand = qv_base_shape
    expected_qv_reshape = (batch * heads, tokens, head_dim)
    expected_k_expand = k_base_shape
    expected_k_reshape = (batch * heads, head_dim, tokens)

    actual_shapes = (
        _resolve_view_shape(shape2, batch * heads * tokens * head_dim),
        _resolve_view_shape(shape3, batch * heads * tokens * head_dim),
        _resolve_view_shape(shape4, batch * heads * tokens * head_dim),
        _resolve_view_shape(shape5, batch * heads * tokens * head_dim),
        _resolve_view_shape(shape6, batch * heads * tokens * head_dim),
        _resolve_view_shape(shape7, batch * heads * tokens * head_dim),
    )
    expected_shapes = (
        expected_qv_expand,
        expected_qv_reshape,
        expected_k_expand,
        expected_k_reshape,
        expected_qv_expand,
        expected_qv_reshape,
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape parameters: {actual_shapes}")

    ynumel = batch * heads * tokens
    return addmm_92, qv_base_shape, k_base_shape, ynumel, tokens, heads, head_dim


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

    addmm_92, qv_shape, k_shape, ynumel, tokens, heads, head_dim = _validate_layout(inputs)
    v_base = torch.empty(qv_shape, device=addmm_92.device, dtype=addmm_92.dtype)
    q_base = torch.empty(qv_shape, device=addmm_92.device, dtype=addmm_92.dtype)
    k_base = torch.empty(k_shape, device=addmm_92.device, dtype=addmm_92.dtype)

    grid = lambda meta: (triton.cdiv(ynumel, meta["YBLOCK"]),)
    _qv_split_layout_kernel[grid](
        addmm_92,
        v_base,
        q_base,
        YNUMEL=ynumel,
        TOKENS=tokens,
        HEADS=heads,
        HEAD_DIM=head_dim,
        SCALE_VALUE=SCALE,
    )
    n_elements = qv_shape[0] * qv_shape[1] * qv_shape[2] * qv_shape[3]
    k_grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _k_split_layout_kernel[k_grid](
        addmm_92,
        k_base,
        N=n_elements,
        TOKENS=tokens,
        HEADS=heads,
        HEAD_DIM=head_dim,
    )
    return (
        v_base.reshape(heads * qv_shape[0], tokens, head_dim).permute(0, 2, 1),
        q_base.reshape(heads * qv_shape[0], tokens, head_dim).permute(0, 2, 1),
        k_base.reshape(heads * k_shape[0], head_dim, tokens).permute(0, 2, 1),
    )


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
