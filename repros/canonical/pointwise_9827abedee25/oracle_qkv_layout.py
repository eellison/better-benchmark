"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin QKV split scope, including the `[B, S, 3, H, D]` view/unbind, Q scaling, three required clone materializations, and the exact returned non-contiguous view layouts with distinct fresh storages, by writing the Q/K/V backing tensors directly in one Triton pass, whereas Inductor lowers the same metadata-heavy layout graph through generic pointwise/layout-copy scheduling; Inductor cannot materially improve this isolated repro today because the required work is dominated by reading each Q/K/V element, writing three fresh outputs, and performing transpose-layout memory movement with only launch-level overhead left; the fix is BANDWIDTH_BOUND: record this case as at floor unless broader layout-copy bandwidth, launch overhead, or multi-output copy scheduling improvements move both paths."""
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


Q_SCALE = 0.1767766952966369


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
            triton.Config({"YBLOCK": 4, "XBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 8, "XBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 64}, num_warps=8, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 32}, num_warps=8, num_stages=3),
        ],
        key=["YNUMEL", "S", "H", "D"],
    )
    @triton.jit
    def _qkv_layout_kernel(
        input_ptr,
        q_base_ptr,
        k_base_ptr,
        v_base_ptr,
        YNUMEL: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        YBLOCK: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        yindex = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
        dindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
        mask = (yindex < YNUMEL) & (dindex < D)

        seq = yindex % S
        batch_head = yindex // S
        head = batch_head % H
        batch = batch_head // H

        qkv_plane = H * D
        input_base = ((batch * S + seq) * 3 * qkv_plane) + head * D + dindex
        q_values = tl.load(input_ptr + input_base, mask=mask, other=0.0)
        k_values = tl.load(input_ptr + input_base + qkv_plane, mask=mask, other=0.0)
        v_values = tl.load(input_ptr + input_base + 2 * qkv_plane, mask=mask, other=0.0)

        qv_base_offsets = yindex * D + dindex
        k_base_offsets = (batch_head * D + dindex) * S + seq
        tl.store(q_base_ptr + qv_base_offsets, q_values * SCALE, mask=mask)
        tl.store(k_base_ptr + tl.trans(k_base_offsets), tl.trans(k_values), mask=tl.trans(mask))
        tl.store(v_base_ptr + qv_base_offsets, v_values, mask=mask)


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
) -> tuple[
    torch.Tensor,
    tuple[int, int, int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
    int,
    int,
    int,
    int,
]:
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

    batch, seq, hidden = _resolve_view_shape(shape0, int(addmm_92.numel()))
    if (batch * seq, hidden) != tuple(addmm_92.shape):
        raise ValueError(
            f"first view shape {(batch, seq, hidden)} does not match input "
            f"shape={tuple(addmm_92.shape)}"
        )

    batch1, seq1, qkv, heads, head_dim = _resolve_view_shape(shape1, int(addmm_92.numel()))
    if (batch1, seq1) != (batch, seq):
        raise ValueError(f"QKV view batch/sequence {(batch1, seq1)} does not match {(batch, seq)}")
    if qkv != 3:
        raise ValueError(f"{REPRO_ID} expects QKV dimension 3, got {qkv}")
    if heads * head_dim * qkv != hidden:
        raise ValueError(
            f"QKV view {(batch1, seq1, qkv, heads, head_dim)} is incompatible "
            f"with hidden={hidden}"
        )

    slice_numel = batch * heads * seq * head_dim
    if slice_numel * 3 != int(addmm_92.numel()):
        raise ValueError("QKV slice size does not match input numel")

    q_base_shape = _resolve_view_shape(shape2, slice_numel)
    q_view_shape = _resolve_view_shape(shape3, slice_numel)
    k_base_shape = _resolve_view_shape(shape4, slice_numel)
    k_view_shape = _resolve_view_shape(shape5, slice_numel)
    v_base_shape = _resolve_view_shape(shape6, slice_numel)
    v_view_shape = _resolve_view_shape(shape7, slice_numel)

    expected_qv_base = (batch, heads, seq, head_dim)
    expected_qv_view = (batch * heads, seq, head_dim)
    expected_k_base = (batch, heads, head_dim, seq)
    expected_k_view = (batch * heads, head_dim, seq)
    if q_base_shape != expected_qv_base or v_base_shape != expected_qv_base:
        raise ValueError(f"unexpected Q/V base shapes {q_base_shape} and {v_base_shape}")
    if q_view_shape != expected_qv_view or v_view_shape != expected_qv_view:
        raise ValueError(f"unexpected Q/V view shapes {q_view_shape} and {v_view_shape}")
    if k_base_shape != expected_k_base:
        raise ValueError(f"unexpected K base shape {k_base_shape}")
    if k_view_shape != expected_k_view:
        raise ValueError(f"unexpected K view shape {k_view_shape}")

    return (
        addmm_92,
        q_base_shape,
        q_view_shape,
        k_base_shape,
        k_view_shape,
        v_base_shape,
        v_view_shape,
        batch,
        seq,
        heads,
        head_dim,
    )


@oracle_impl(hardware="H100", shapes="(T([6272, 3072], f32), S([128, 49, 3072]), S([128, 49, 3, 32, -1]), S([128, 32, 49, 32]), S([4096, 49, 32]), S([128, 32, 32, 49]), S([4096, 32, 49]), S([128, 32, 49, 32]), S([4096, 49, 32]))")
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

    (
        addmm_92,
        q_base_shape,
        q_view_shape,
        k_base_shape,
        k_view_shape,
        v_base_shape,
        v_view_shape,
        batch,
        seq,
        heads,
        head_dim,
    ) = _validate_layout(inputs)

    q_base = torch.empty(q_base_shape, device=addmm_92.device, dtype=addmm_92.dtype)
    k_base = torch.empty(k_base_shape, device=addmm_92.device, dtype=addmm_92.dtype)
    v_base = torch.empty(v_base_shape, device=addmm_92.device, dtype=addmm_92.dtype)

    y_elements = batch * heads * seq
    grid = lambda meta: (triton.cdiv(head_dim, meta["XBLOCK"]), triton.cdiv(y_elements, meta["YBLOCK"]))
    _qkv_layout_kernel[grid](
        addmm_92,
        q_base,
        k_base,
        v_base,
        YNUMEL=y_elements,
        S=seq,
        H=heads,
        D=head_dim,
        SCALE=Q_SCALE,
    )
    return (
        v_base.view(v_view_shape).permute(0, 2, 1),
        q_base.view(q_view_shape).permute(0, 2, 1),
        k_base.view(k_view_shape).permute(0, 2, 1),
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
