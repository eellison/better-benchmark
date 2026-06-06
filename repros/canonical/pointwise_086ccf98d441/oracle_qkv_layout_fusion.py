"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete VisFormer QKV layout split returned by Repro.forward, including the initial `[B, 3, H, D, P]` view, unbind into Q/K/V, the three required clone materializations, and the final permuted view outputs with eager-compatible strides and non-aliasing fresh storages, by grouping the identical Q/V layout copies in one Triton kernel and emitting the K contiguous copy in a second kernel, whereas Inductor emits three separate specialized layout-copy kernels for the same metadata-only view/permute/unbind/expand/clone/view/permute graph; Inductor cannot materially improve this local scope today because the remaining work is dominated by mandatory Q/K/V input reads, three fresh output writes, and transpose-layout memory movement rather than avoidable arithmetic or intermediate tensors; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader layout-copy bandwidth, launch-overhead, or multi-output copy scheduling work moves both implementations beyond the current CUDAGraph-measured envelope."""
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"YBLOCK": 8, "XBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 128}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 64}, num_warps=8, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 128}, num_warps=8, num_stages=3),
        ],
        key=["YNUMEL", "D", "HEADS", "P"],
    )
    @triton.jit
    def _qv_layout_fusion_kernel(
        input_ptr,
        q_base_ptr,
        v_base_ptr,
        YNUMEL: tl.constexpr,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
        YBLOCK: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        yindex = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
        mask = (yindex < YNUMEL) & (xindex < D)

        pos = yindex % P
        head = (yindex // P) % HEADS
        batch = yindex // (P * HEADS)

        qkv_plane = HEADS * D * P
        q_input_offsets = batch * (3 * qkv_plane) + (head * D + xindex) * P + pos
        v_input_offsets = q_input_offsets + 2 * qkv_plane
        output_offsets = xindex + D * yindex

        q_values = tl.load(input_ptr + q_input_offsets, mask=mask, other=0.0)
        v_values = tl.load(input_ptr + v_input_offsets, mask=mask, other=0.0)
        tl.store(q_base_ptr + output_offsets, q_values, mask=mask)
        tl.store(v_base_ptr + output_offsets, v_values, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        ],
        key=["N", "HEADS", "D", "P"],
    )
    @triton.jit
    def _k_layout_copy_kernel(
        input_ptr,
        k_base_ptr,
        N: tl.constexpr,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        qkv_plane = HEADS * D * P
        within_batch = offsets % qkv_plane
        batch = offsets // qkv_plane
        values = tl.load(input_ptr + qkv_plane + within_batch + batch * (3 * qkv_plane), mask=mask, other=0.0)
        tl.store(k_base_ptr + offsets, values, mask=mask)


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


def _validate_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], int, int, int, int]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects eight inputs, got {len(inputs)}")

    convolution_53, shape0, shape1, shape2, shape3, shape4, shape5, shape6 = inputs
    if not isinstance(convolution_53, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not convolution_53.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if convolution_53.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {convolution_53.dtype}")
    if not convolution_53.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(convolution_53.stride())}")

    b, qkv, heads, d_model, p_flat = _resolve_view_shape(shape0, int(convolution_53.numel()))
    if qkv != 3:
        raise ValueError(f"{REPRO_ID} expects a QKV dimension of 3, got {qkv}")

    slice_numel = b * heads * d_model * p_flat
    if slice_numel * 3 != int(convolution_53.numel()):
        raise ValueError("QKV slice size does not match input numel")

    q_expand_shape = _resolve_view_shape(shape1, slice_numel)
    q_base_shape = _resolve_view_shape(shape2, slice_numel)
    k_expand_shape = _resolve_view_shape(shape3, slice_numel)
    k_base_shape = _resolve_view_shape(shape4, slice_numel)
    v_expand_shape = _resolve_view_shape(shape5, slice_numel)
    v_base_shape = _resolve_view_shape(shape6, slice_numel)

    expected_qv_expand = (b, heads, p_flat, d_model)
    expected_k_expand = (b, heads, d_model, p_flat)
    expected_qv_base = (b * heads, p_flat, d_model)
    expected_k_base = (b * heads, d_model, p_flat)
    if q_expand_shape != expected_qv_expand or v_expand_shape != expected_qv_expand:
        raise ValueError(f"unexpected Q/V expand shapes {q_expand_shape} and {v_expand_shape}")
    if k_expand_shape != expected_k_expand:
        raise ValueError(f"unexpected K expand shape {k_expand_shape}")
    if q_base_shape != expected_qv_base or v_base_shape != expected_qv_base:
        raise ValueError(f"unexpected Q/V base shapes {q_base_shape} and {v_base_shape}")
    if k_base_shape != expected_k_base:
        raise ValueError(f"unexpected K base shape {k_base_shape}")

    return convolution_53, q_base_shape, k_base_shape, v_base_shape, b, heads, d_model, p_flat


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

    convolution_53, q_shape, k_shape, v_shape, b, heads, d_model, p_flat = _validate_layout(inputs)
    q_base = torch.empty(q_shape, device=convolution_53.device, dtype=convolution_53.dtype)
    k_base = torch.empty(k_shape, device=convolution_53.device, dtype=convolution_53.dtype)
    v_base = torch.empty(v_shape, device=convolution_53.device, dtype=convolution_53.dtype)

    n_elements = b * heads * d_model * p_flat
    y_elements = b * heads * p_flat
    qv_grid = lambda meta: (triton.cdiv(d_model, meta["XBLOCK"]), triton.cdiv(y_elements, meta["YBLOCK"]))
    _qv_layout_fusion_kernel[qv_grid](
        convolution_53,
        q_base,
        v_base,
        YNUMEL=y_elements,
        HEADS=heads,
        D=d_model,
        P=p_flat,
    )
    k_grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _k_layout_copy_kernel[k_grid](
        convolution_53,
        k_base,
        N=n_elements,
        HEADS=heads,
        D=d_model,
        P=p_flat,
    )
    return (v_base.permute(0, 2, 1), q_base.permute(0, 2, 1), k_base.permute(0, 2, 1))


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
