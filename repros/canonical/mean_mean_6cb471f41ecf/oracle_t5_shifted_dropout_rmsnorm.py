"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5 token embedding gather plus shifted decoder-token construction, two independent Inductor-seeded dropout masks from seed indices 0 and 26 of 64 generated seeds, two fp32 hidden-size-512 RMSNorm mean(square)+rsqrt eps=1e-6 branches, separate affine weight multiplies, and the two returned [8192,512] views in one shape-specialized Triton row kernel; exact stochastic value equality with the eager repro is not established, so this is a full-scope structural oracle but true_floor=no, whereas Inductor lowers the same graph through generic embedding, shift/scatter/where, RNG/dropout, reduction, pointwise, and view scheduling paths; Inductor cannot do this today because its normalization/pattern scheduler does not recognize token and shifted-token embedding producers feeding sibling stochastic RMSNorm consumers as one fused template; the fix is NEW_PATTERN: add a guarded T5 embedding-shift-dual-dropout-RMSNorm lowering that folds token shifting, indexed gathers, RNG domains, fixed-K RMS reductions, affine epilogues, and sibling view stores into specialized codegen."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops

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

BATCH = 8
SEQ_LEN = 1024
ROWS = BATCH * SEQ_LEN
HIDDEN = 512
VOCAB = 32128
TOKEN_SHAPE = (BATCH, SEQ_LEN)
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
VIEW_SHAPE = (ROWS, HIDDEN)
EPS = 1.0e-6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 64
SEED_INDEX_0 = 0
SEED_INDEX_1 = 26
IGNORE_INDEX = -100
BLOCK_M = 2
BLOCK_H = 512
EXACT_STOCHASTIC_EQUALITY = False
TRUE_FLOOR = False

if triton is not None:

    @triton.jit
    def _t5_shifted_dual_dropout_rmsnorm_kernel(
        embedding_ptr,
        input_ids_ptr,
        weight0_ptr,
        shifted_ids_ptr,
        weight1_ptr,
        seeds_ptr,
        out0_ptr,
        out1_ptr,
        rows_total: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        seed_index0: tl.constexpr,
        seed_index1: tl.constexpr,
        ignore_index: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_h)

        batch = rows // seq_len
        seq = rows - batch * seq_len

        token0 = tl.load(input_ids_ptr + rows)
        prev_pos = batch * seq_len + seq - 1
        offsets = rows[:, None] * hidden + cols[None, :]
        embedding0 = tl.load(
            embedding_ptr + token0[:, None] * hidden + cols[None, :],
        ).to(tl.float32)
        seed0 = tl.load(seeds_ptr + seed_index0)
        rand0 = tl.rand(seed0, offsets.to(tl.uint32))
        dropped0 = tl.where(rand0 > dropout_p, embedding0 * dropout_scale, 0.0)
        square_sum0 = tl.sum(dropped0 * dropped0, axis=1)
        inv_rms0 = tl.rsqrt(square_sum0 / hidden + eps)
        weight0 = tl.load(weight0_ptr + cols).to(tl.float32)
        result0 = dropped0 * inv_rms0[:, None] * weight0[None, :]
        tl.store(out0_ptr + offsets, result0)

        token1 = tl.load(shifted_ids_ptr + prev_pos, mask=seq != 0, other=0)
        token1 = tl.where(token1 == ignore_index, 0, token1)
        embedding1 = tl.load(
            embedding_ptr + token1[:, None] * hidden + cols[None, :],
        ).to(tl.float32)
        seed1 = tl.load(seeds_ptr + seed_index1)
        rand1 = tl.rand(seed1, offsets.to(tl.uint32))
        dropped1 = tl.where(rand1 > dropout_p, embedding1 * dropout_scale, 0.0)
        square_sum1 = tl.sum(dropped1 * dropped1, axis=1)
        inv_rms1 = tl.rsqrt(square_sum1 / hidden + eps)
        weight1 = tl.load(weight1_ptr + cols).to(tl.float32)
        result1 = dropped1 * inv_rms1[:, None] * weight1[None, :]
        tl.store(out1_ptr + offsets, result1)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    embedding, input_ids, weight0, shifted_ids, weight1, shape0, shape1 = inputs
    embedding_t = _require_tensor("arg1_1", embedding, (VOCAB, HIDDEN), torch.float32)
    input_ids_t = _require_tensor("arg0_1", input_ids, TOKEN_SHAPE, torch.int64)
    weight0_t = _require_tensor("arg2_1", weight0, (HIDDEN,), torch.float32)
    shifted_ids_t = _require_tensor("arg52_1", shifted_ids, TOKEN_SHAPE, torch.int64)
    weight1_t = _require_tensor("arg53_1", weight1, (HIDDEN,), torch.float32)

    if (
        input_ids_t.device != embedding_t.device
        or weight0_t.device != embedding_t.device
        or shifted_ids_t.device != embedding_t.device
        or weight1_t.device != embedding_t.device
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    out_shape0 = _require_shape("_shape_param_0", shape0, VIEW_SHAPE)
    out_shape1 = _require_shape("_shape_param_1", shape1, VIEW_SHAPE)
    return embedding_t, input_ids_t, weight0_t, shifted_ids_t, weight1_t, out_shape0, out_shape1


def _make_inductor_seeds(device: torch.device) -> torch.Tensor:
    return torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)


@oracle_impl(hardware="H100", shapes="(T([32128, 512], f32), T([8, 1024], i64, gen=Index(32128)), T([512], f32), T([8, 1024], i64), T([512], f32), S([8192, 512]), S([8192, 512]))")
def oracle_forward(inputs):
    """Run the complete T5 shifted-token embedding + dual dropout + RMSNorm scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same two f32 [8192,512] view outputs. The second branch folds the exact
    decoder-token shift and -100 replacement before the embedding gather.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_t5_shifted_dropout_rmsnorm.py")

    embedding, input_ids, weight0, shifted_ids, weight1, shape0, shape1 = _validate_inputs(inputs)
    out0 = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=embedding.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=embedding.device,
        dtype=torch.float32,
    )
    seeds = _make_inductor_seeds(embedding.device)

    _t5_shifted_dual_dropout_rmsnorm_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        embedding,
        input_ids,
        weight0,
        shifted_ids,
        weight1,
        seeds,
        out0,
        out1,
        rows_total=ROWS,
        seq_len=SEQ_LEN,
        hidden=HIDDEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        seed_index0=SEED_INDEX_0,
        seed_index1=SEED_INDEX_1,
        ignore_index=IGNORE_INDEX,
        block_m=BLOCK_M,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=1,
    )
    return (out0.view(shape0), out1.view(shape1))


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
