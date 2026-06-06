"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Visformer training QKV layout assembly by reading the three BMM outputs and writing the final contiguous `[B, 3*H*D, S, S]` image layout directly, whereas Inductor already emits an equivalent single fused layout-copy kernel for the reshape/permute/cat/reshape/permute/clone/reshape chain; Inductor cannot materially improve this case through a local scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, recompute-fusion, or new-pattern change because the remaining work is dominated by the mandatory Q/K/V reads, transposed layout stores, and launch overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope oracle unless broader layout-copy bandwidth or launch-overhead work moves both paths."""
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
            triton.Config({"YBLOCK": 8, "XBLOCK": 128}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 128}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 64}, num_warps=8, num_stages=3),
            triton.Config({"YBLOCK": 64, "XBLOCK": 32}, num_warps=8, num_stages=3),
        ],
        key=["YNUMEL", "P", "D", "HEADS", "B"],
    )
    @triton.jit
    def _qkv_image_layout_kernel(
        bmm_v_ptr,
        bmm_k_ptr,
        bmm_q_ptr,
        out_ptr,
        B: tl.constexpr,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
        YNUMEL: tl.constexpr,
        YBLOCK: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
        x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]

        d = y % D
        head = (y // D) % HEADS
        batch_qkv = y // (HEADS * D)
        batch = batch_qkv % B
        qkv = batch_qkv // B
        channel = y % (HEADS * D)

        qv_input_offsets = d + D * x + (P * D) * head + (HEADS * P * D) * batch
        k_input_offsets = x + P * d + (P * D) * head + (HEADS * P * D) * batch
        output_offsets = x + P * channel + (HEADS * D * P) * qkv + (3 * HEADS * D * P) * batch

        mask = (y < YNUMEL) & (x < P)
        v_values = tl.load(bmm_v_ptr + qv_input_offsets, mask=mask & (qkv == 0), other=0.0)
        k_values = tl.load(bmm_k_ptr + k_input_offsets, mask=mask & (qkv == 1), other=0.0)
        q_values = tl.load(bmm_q_ptr + qv_input_offsets, mask=mask & (qkv == 2), other=0.0)
        values = tl.where(qkv == 0, v_values, tl.where(qkv == 1, k_values, q_values))
        tl.store(out_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _expect_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    bmm_44, bmm_46, bmm_47, shape0, shape1, shape2, shape3, shape4 = inputs
    for name, tensor in (("bmm_44", bmm_44), ("bmm_46", bmm_46), ("bmm_47", bmm_47)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be torch.float32, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tuple(tensor.stride())}")

    view_v = _shape_tuple(shape0, "_shape_param_0")
    view_k = _shape_tuple(shape1, "_shape_param_1")
    view_q = _shape_tuple(shape2, "_shape_param_2")
    cat_view = _shape_tuple(shape3, "_shape_param_3")
    output_shape = _shape_tuple(shape4, "_shape_param_4")

    if len(view_v) != 4 or len(view_k) != 4 or len(view_q) != 4:
        raise ValueError(f"unexpected input view shapes: {view_v}, {view_k}, {view_q}")
    batch, heads, positions, dim = view_v
    if view_q != view_v:
        raise ValueError(f"Q/V view shapes must match, got {view_q} and {view_v}")
    if view_k != (batch, heads, dim, positions):
        raise ValueError(f"unexpected K view shape {view_k}, expected {(batch, heads, dim, positions)}")
    if cat_view != (3, batch, heads, positions, dim):
        raise ValueError(f"unexpected cat view shape {cat_view}")

    expected_bmm_qv = (batch * heads, positions, dim)
    expected_bmm_k = (batch * heads, dim, positions)
    if tuple(bmm_44.shape) != expected_bmm_qv:
        raise ValueError(f"unexpected bmm_44 shape {tuple(bmm_44.shape)}")
    if tuple(bmm_47.shape) != expected_bmm_qv:
        raise ValueError(f"unexpected bmm_47 shape {tuple(bmm_47.shape)}")
    if tuple(bmm_46.shape) != expected_bmm_k:
        raise ValueError(f"unexpected bmm_46 shape {tuple(bmm_46.shape)}")

    side = int(positions**0.5)
    if side * side != positions:
        raise ValueError(f"positions must be a square image, got {positions}")
    expected_output = (batch, 3 * heads * dim, side, side)
    if output_shape != expected_output:
        raise ValueError(f"unexpected output shape {output_shape}, expected {expected_output}")

    return bmm_44, bmm_46, bmm_47, output_shape, batch, heads, dim, positions


def oracle_forward(inputs):
    """Run the complete Repro.forward layout scope and return the final image tensor."""
    bmm_44, bmm_46, bmm_47, output_shape, batch, heads, dim, positions = _expect_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=bmm_44.device,
        dtype=bmm_44.dtype,
    )
    grid = lambda meta: (
        triton.cdiv(positions, meta["XBLOCK"]),
        triton.cdiv(3 * batch * heads * dim, meta["YBLOCK"]),
    )
    _qkv_image_layout_kernel[grid](
        bmm_47,
        bmm_46,
        bmm_44,
        output,
        B=batch,
        HEADS=heads,
        D=dim,
        P=positions,
        YNUMEL=3 * batch * heads * dim,
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
