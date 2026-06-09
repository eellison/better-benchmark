"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete ViT inference residual add, fp32 hidden-size-384 LayerNorm with eps=1e-6 and fp16 affine inputs, final fp16 cast, and CLS-token select/contiguous clone by reducing only the 32 token-0 rows that are observable in Repro.forward, whereas Inductor lowers the decomposed view/add/var_mean/affine/cast/select/clone graph as a full [32,197,384] row-normalization producer before the trailing select; Inductor cannot do this today because its scheduler does not push the token-index select backward through row-independent residual-add plus LayerNorm reductions to shrink the row iteration domain from 6304 rows to 32 rows; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to sink constant token selects through row-local normalization graphs and generate the narrowed residual-add/LayerNorm loop nest directly."""
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
BATCH = 32
TOKENS = 197
HIDDEN = 384
ROWS = BATCH * TOKENS
ADDMM_SHAPE = (ROWS, HIDDEN)
ADD_SHAPE = (BATCH, TOKENS, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (BATCH, HIDDEN)
EPS = 1.0e-6

if triton is not None:

    @triton.jit
    def _vit_layernorm_cls_kernel(
        addmm_ptr,
        add_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        row = batch * tokens
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(add_ptr + offsets, mask=mask, other=0.0)
        x = (addmm + residual).to(tl.float16).to(tl.float32)
        x = tl.where(mask, x, 0.0)

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias
        tl.store(out_ptr + batch * hidden + cols, y, mask=mask)


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_47, add_80, arg149_1, arg150_1, shape_param = inputs
    tensor_inputs = (addmm_47, add_80, arg149_1, arg150_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (ADDMM_SHAPE, ADD_SHAPE, AFFINE_SHAPE, AFFINE_SHAPE)
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape_param) != ADD_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    if not (
        add_80.device == addmm_47.device
        and arg149_1.device == addmm_47.device
        and arg150_1.device == addmm_47.device
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_47, add_80, arg149_1, arg150_1


@oracle_impl(hardware="H100", shapes="(T([6304, 384], f16), T([32, 197, 384], f16), T([384], f16), T([384], f16), S([32, 197, 384]))")
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
        raise RuntimeError("Triton is required for oracle_vit_layernorm_cls.py")

    addmm_47, add_80, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        (HIDDEN, 1),
        device=addmm_47.device,
        dtype=torch.float16,
    )
    _vit_layernorm_cls_kernel[(BATCH,)](
        addmm_47,
        add_80,
        weight,
        bias,
        output,
        tokens=TOKENS,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=512,
        num_warps=1,
        num_stages=3,
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
