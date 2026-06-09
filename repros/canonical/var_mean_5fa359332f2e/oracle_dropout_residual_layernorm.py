"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BART training residual-dropout LayerNorm scope in one shape-specialized Triton row kernel, including the `[2048, 768] -> [4, 512, 768]` view, Inductor-seeded dropout on the projected residual, residual add, fp32 hidden-size-768 `var_mean`, eps=1e-5 affine epilogue, final `[2048, 768]` view, and `rsqrt / 768` side output, whereas tuned Inductor already measures within the same mandatory memory-traffic and launch envelope for this captured norm-template graph; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining work is the required projected/residual/affine reads, row reduction, RNG/dropout, output write, side-output store, and launch overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor structural oracle unless a broader RNG or normalization-template improvement moves both paths. Exact stochastic activation equality is not established by the harness, so this should be marked not_true_floor rather than a true exact floor proof."""
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
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 4
SEQ_LEN = 512
HIDDEN = 768
N_ROWS = BATCH * SEQ_LEN
N_SEEDS = 5
SEED_INDEX = 2
EPS = 1.0e-5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
BLOCK_H = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _dropout_residual_layernorm_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        seed_index: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        col_mask = cols < hidden
        offsets = row * hidden + cols

        projected = tl.load(addmm_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        random_offsets = offsets.to(tl.uint32)
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, random_offsets)

        dropped = tl.where(random > dropout_p, projected * dropout_scale, 0.0)
        x = residual + dropped
        x_for_reduce = tl.where(col_mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        centered_for_var = tl.where(col_mask, centered, 0.0)
        variance = tl.sum(centered_for_var * centered_for_var, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=col_mask)
        tl.store(invstd_div_ptr + row, invstd / hidden)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm, seeds, residual, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (addmm, seeds, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (N_ROWS, HIDDEN),
        (N_SEEDS,),
        (BATCH, SEQ_LEN, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")
        if value.device != addmm.device:
            raise ValueError(f"input {index} device {value.device} != {addmm.device}")

    if addmm.dtype != torch.float32 or residual.dtype != torch.float32:
        raise TypeError("activation inputs must be torch.float32")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("layernorm weight and bias must be torch.float32")
    if seeds.dtype != torch.int64:
        raise TypeError(f"Inductor seeds must be torch.int64, got {seeds.dtype}")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if shape0_tuple != (BATCH, SEQ_LEN, HIDDEN):
        raise ValueError(f"unexpected input view shape parameter: {shape0!r}")
    if shape1_tuple != (N_ROWS, HIDDEN):
        raise ValueError(f"unexpected output view shape parameter: {shape1!r}")

    return addmm, seeds, residual, weight, bias, shape0_tuple, shape1_tuple


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, seeds, residual, weight, bias, shape0, shape1 = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm, shape0)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape0, seed, "rand")
    dropped = torch.ops.aten.mul.Tensor(torch.ops.aten.gt.Scalar(random, DROPOUT_P), view_default)
    dropped = torch.ops.aten.mul.Tensor(dropped, DROPOUT_SCALE)
    add_tensor = torch.ops.aten.add.Tensor(residual, dropped)
    var, mean = torch.ops.aten.var_mean.correction(add_tensor, [2], correction=0, keepdim=True)
    invstd = torch.ops.aten.rsqrt.default(var + EPS)
    normalized = (add_tensor - mean) * invstd
    affine = normalized * weight + bias
    return torch.ops.aten.view.default(affine, shape1), invstd / HIDDEN


@oracle_impl(hardware="H100", shapes="(T([2048, 768], f32), T([5], i64), T([4, 512, 768], f32), T([768], f32), T([768], f32), S([4, 512, 768]), S([2048, 768]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope: seeded dropout, residual LayerNorm, affine, and invstd side output."""
    addmm, seeds, residual, weight, bias, _shape0, shape1 = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        shape1,
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    _dropout_residual_layernorm_kernel[(N_ROWS,)](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        invstd_div,
        hidden=HIDDEN,
        seed_index=SEED_INDEX,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        block_h=BLOCK_H,
        num_warps=1,
        num_stages=3,
    )
    return output, invstd_div


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
