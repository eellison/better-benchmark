"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ALBERT training attention-softmax region returned by Repro.forward, including the generated arange/ge mask, zero/-inf where bias, [512,512,512] -> [8,64,512,512] view, bias add, stable last-dimension softmax, any(eq(-inf)) all-masked-row zero fallback, expand, and final contiguous [512,512,512] view, by proving the generated predicate always selects the zero bias for this captured shape and folding the live semantics into one row-softmax Triton kernel; whereas Inductor lowers the decomposed iota/where/add/amax/sub/exp/sum/div/eq/any/where/expand/view graph through its compiled softmax path; Inductor is already at the CUDAGraph-measured floor for this shape because the required 512-wide row softmax over 262144 rows is dominated by input/output bandwidth and exp/reduction work after dispatch overhead is removed; the fix is BANDWIDTH_BOUND: record this row as at floor, with only optional shape-derived mask cleanup for graph simplification rather than a material local performance fix."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

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
    has_stochastic_ops,
)


BATCH = 8
HEADS = 64
Q_LEN = 512
K_LEN = 512
INPUT_SHAPE = (512, 512, 512)
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
EXPAND_MASK_SHAPE = (BATCH, -1, Q_LEN, K_LEN)
N_ROWS = BATCH * HEADS * Q_LEN
OUT_SHAPE = INPUT_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> None:
    if not isinstance(bmm, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(bmm).__name__}")
    if not bmm.is_cuda:
        raise RuntimeError("CUDA tensor input is required")
    if bmm.dtype != torch.float32:
        raise TypeError(f"expected bmm dtype torch.float32, got {bmm.dtype}")
    if tuple(bmm.shape) != INPUT_SHAPE:
        raise ValueError(f"expected bmm shape {INPUT_SHAPE}, got {tuple(bmm.shape)}")
    if tuple(bmm.stride()) != OUT_STRIDE:
        raise ValueError(f"expected contiguous bmm stride {OUT_STRIDE}, got {tuple(bmm.stride())}")

    _validate_shape_param("_shape_param_0", _shape_param_0, EXPAND_MASK_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, VIEW_SHAPE)
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def _aten_fallback(
    bmm: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> torch.Tensor:
    iota_default = torch.ops.prims.iota.default(
        512,
        start=0,
        step=1,
        dtype=torch.int64,
        device=bmm.device,
        requires_grad=False,
    )
    add_tensor = torch.ops.aten.add.Tensor(iota_default, 0)
    unsqueeze_default = torch.ops.aten.unsqueeze.default(add_tensor, 0)
    unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1)
    unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3)
    ge_scalar = torch.ops.aten.ge.Scalar(unsqueeze_default_2, 0)
    expand_default = torch.ops.aten.expand.default(ge_scalar, _shape_param_0)
    full_default = torch.ops.aten.full.default(
        [],
        -float("inf"),
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm.device,
        pin_memory=False,
    )
    full_default_1 = torch.ops.aten.full.default(
        [],
        0.0,
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm.device,
        pin_memory=False,
    )
    where_self = torch.ops.aten.where.self(expand_default, full_default_1, full_default)
    view_default = torch.ops.aten.view.default(bmm, _shape_param_1)
    add_tensor_1 = torch.ops.aten.add.Tensor(view_default, where_self)
    amax_default = torch.ops.aten.amax.default(add_tensor_1, [-1], True)
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, amax_default)
    exp_default = torch.ops.aten.exp.default(sub_tensor)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list)
    eq_scalar = torch.ops.aten.eq.Scalar(add_tensor_1, -float("inf"))
    logical_not_default = torch.ops.aten.logical_not.default(eq_scalar)
    any_dim = torch.ops.aten.any.dim(logical_not_default, -1, True)
    logical_not_default_1 = torch.ops.aten.logical_not.default(any_dim)
    full_default_2 = torch.ops.aten.full.default(
        [BATCH, HEADS, Q_LEN, K_LEN],
        0,
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm.device,
        pin_memory=False,
    )
    where_self_1 = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor)
    expand_default_1 = torch.ops.aten.expand.default(where_self_1, _shape_param_2)
    return torch.ops.aten.view.default(expand_default_1, _shape_param_3)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _all_inf_safe_softmax_kernel(
        bmm_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        k_len: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_k)
        offsets = rows[:, None] * k_len + cols[None, :]
        mask = rows[:, None] < n_rows

        # The generated arange(512) >= 0 mask is true for every query position,
        # so the broadcast where contributes exactly zero. The all--inf guard is
        # still live and maps such rows to all zeros.
        scores = tl.load(bmm_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(scores, axis=1)
        has_any = row_max != -float("inf")
        stable_max = tl.where(has_any, row_max, 0.0)
        numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
        denom = tl.sum(numer, axis=1)
        probs = tl.where(has_any[:, None], numer / denom[:, None], 0.0)

        tl.store(out_ptr + offsets, probs, mask=mask)


def oracle_full_attention_softmax(
    bmm: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
    *,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 2,
) -> torch.Tensor:
    _validate_inputs(
        bmm,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    if triton is None:
        return _aten_fallback(
            bmm,
            _shape_param_0,
            _shape_param_1,
            _shape_param_2,
            _shape_param_3,
        )
    if not bmm.is_contiguous():
        return _aten_fallback(
            bmm,
            _shape_param_0,
            _shape_param_1,
            _shape_param_2,
            _shape_param_3,
        )
    if block_k != K_LEN:
        raise ValueError(f"block_k must equal {K_LEN} for this fixed-shape oracle, got {block_k}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")

    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(N_ROWS, block_m),)
    _all_inf_safe_softmax_kernel[grid](
        bmm,
        output,
        n_rows=N_ROWS,
        k_len=K_LEN,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return output


@oracle_impl(hardware="H100", shapes="(T([512, 512, 512], f32), S([8, -1, 512, 512]), S([8, 64, 512, 512]), S([8, 64, 512, 512]), S([512, 512, 512]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for the assigned input tuple."""
    return oracle_full_attention_softmax(*inputs)


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
