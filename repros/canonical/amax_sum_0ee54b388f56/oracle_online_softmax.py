"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Visformer scaled attention softmax captured in Repro.forward, including the input view, no-op multiply, unscaled row amax, scaled stable exp/sum/div normalization, expand, and final contiguous view in one Triton row kernel, whereas Inductor currently lowers the decomposed amax/sub/mul/exp/sum/div chain through its generic small-reduction schedule; Inductor cannot do this today because its pattern/codegen path does not canonicalize this view-preserving scaled softmax into a dedicated multi-row small-K template; the fix is NEW_PATTERN: add a guarded scaled-attention-softmax lowering that recognizes the paired amax/sum idiom with intervening scale and emits the row-wise template only when it beats the generic schedule."""
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
REPRO_ID = "amax_sum_0ee54b388f56"
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


SCALE = 0.08838834764831845
SCALE_LOG2E = SCALE * 1.4426950408889634
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_scope(
    bmm_14: torch.Tensor,
    shape_param_0,
    shape_param_1,
    shape_param_2,
) -> tuple[tuple[int, int, int], int, int]:
    if not isinstance(bmm_14, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(bmm_14).__name__}")
    if bmm_14.dtype != torch.float32:
        raise TypeError(f"expected fp32 input, got {bmm_14.dtype}")
    if bmm_14.ndim != 3:
        raise ValueError(f"expected rank-3 input, got rank {bmm_14.ndim}")

    input_shape = _shape_tuple(bmm_14.shape)
    view_shape = _shape_tuple(shape_param_0)
    expand_shape = _shape_tuple(shape_param_1)
    output_shape = _shape_tuple(shape_param_2)

    if len(view_shape) != 4:
        raise ValueError(f"expected rank-4 first view shape, got {view_shape}")
    if expand_shape != view_shape:
        raise ValueError(f"expand shape {expand_shape} does not match view shape {view_shape}")
    if output_shape != input_shape:
        raise ValueError(f"final view shape {output_shape} does not match input shape {input_shape}")

    batch, heads, q_len, k_len = view_shape
    if (batch * heads, q_len, k_len) != input_shape:
        raise ValueError(
            f"view shape {view_shape} is not compatible with input shape {input_shape}"
        )
    if k_len <= 0:
        raise ValueError("last dimension must be non-empty")
    if k_len > 256:
        raise ValueError(f"oracle supports K <= 256 for this repro, got {k_len}")

    return input_shape, batch * heads * q_len, k_len


def _aten_fallback(bmm_14, shape_param_0, shape_param_1, shape_param_2):
    view_default = torch.ops.aten.view.default(bmm_14, shape_param_0)
    mul_tensor = torch.ops.aten.mul.Tensor(view_default, 1)
    amax_default = torch.ops.aten.amax.default(mul_tensor, [-1], True)
    sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, SCALE)
    exp_default = torch.ops.aten.exp.default(mul_tensor_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list)
    expand_default = torch.ops.aten.expand.default(div_tensor, shape_param_1)
    return torch.ops.aten.view.default(expand_default, shape_param_2)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=2, num_stages=4),
        ],
        key=["n_rows", "n_cols"],
    )
    @triton.jit
    def oracle_kernel(
        input_ptr,
        output_ptr,
        n_rows: tl.constexpr,
        n_cols: tl.constexpr,
        block_cols: tl.constexpr,
        scale_log2e: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, block_cols)
        mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)
        offsets = row_offsets[:, None] * n_cols + col_offsets[None, :]

        vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(vals, axis=1)
        numer = tl.exp2((vals - row_max[:, None]) * scale_log2e)
        denom = tl.sum(numer, axis=1)
        out_vals = numer / denom[:, None]
        tl.store(output_ptr + offsets, out_vals, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([768, 49, 49], f32), S([128, 6, 49, 49]), S([128, 6, 49, 49]), S([768, 49, 49]))")
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
    bmm_14, shape_param_0, shape_param_1, shape_param_2 = inputs
    output_shape, n_rows, n_cols = _validate_scope(
        bmm_14,
        shape_param_0,
        shape_param_1,
        shape_param_2,
    )

    if triton is None or not bmm_14.is_cuda or not bmm_14.is_contiguous():
        return _aten_fallback(bmm_14, shape_param_0, shape_param_1, shape_param_2)

    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=bmm_14.device,
        dtype=bmm_14.dtype,
    )
    block_cols = triton.next_power_of_2(n_cols)
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)
    oracle_kernel[grid](
        bmm_14,
        output,
        n_rows=n_rows,
        n_cols=n_cols,
        block_cols=block_cols,
        scale_log2e=SCALE_LOG2E,
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
