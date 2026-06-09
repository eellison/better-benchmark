"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT word/token-type/position embedding assembly, fp32 hidden-size-768 population var_mean with eps=1e-12, affine LayerNorm, Inductor-style dropout, final contiguous [16384, 768] view, and live rsqrt/768 side output in one fixed-hidden Triton row kernel, whereas tuned Inductor already reaches the same practical embedding-gather, row-reduction, dropout, output-store, and side-output envelope for this fixed hidden size; Inductor cannot materially improve this repro through a narrower norm-template canonicalization change because the remaining work is dominated by mandatory embedding/affine reads, one row reduction, RNG/dropout, output writes, and launch overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope embedding-layernorm-dropout case unless broader normalization, RNG, or launch-overhead improvements move both paths. Exact stochastic equality with eager is not established when correctness skips the dropout output, so such measurements are not_true_floor."""
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
BATCH = 32
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
WORD_VOCAB = 30522
TOKEN_TYPES = 2
POSITION_VOCAB = 512
EPS = 1.0e-12
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 37
SEED_INDEX = 0
BLOCK_H = 1024
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)

if triton is not None:

    @triton.jit
    def _bert_embedding_layernorm_dropout_kernel(
        token_type_source_ptr,
        position_ids_ptr,
        word_embedding_ptr,
        word_ids_ptr,
        token_type_embedding_ptr,
        position_embedding_ptr,
        weight_ptr,
        bias_ptr,
        seed_ptr,
        out_ptr,
        side_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        seq = row % seq_len
        word_id = tl.load(word_ids_ptr + row)
        position_id = tl.load(position_ids_ptr + seq)
        token_type_id = tl.load(token_type_source_ptr + position_id)

        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        token_type = tl.load(
            token_type_embedding_ptr + token_type_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        x = tl.where(mask, word + token_type + position, 0.0)
        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        offsets = row * hidden + cols
        seed = tl.load(seed_ptr)
        random = tl.rand(seed, offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, y, 0.0) * dropout_scale

        tl.store(out_ptr + offsets, dropped, mask=mask)
        tl.store(side_ptr + row, invstd / hidden)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        token_type_source,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        token_type_expand_shape,
        output_shape,
    ) = inputs

    tensor_inputs = (
        token_type_source,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first eight repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (1, SEQ_LEN),
        (1, SEQ_LEN),
        (WORD_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (TOKEN_TYPES, HIDDEN),
        (POSITION_VOCAB, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    index_tensors = (token_type_source, position_ids, word_ids)
    if not all(value.dtype == torch.int64 for value in index_tensors):
        raise TypeError("token-type source, position ids, and word ids must be torch.int64")
    fp32_tensors = (
        word_embedding,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
    )
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float32")

    token_type_expand_shape_tuple = _shape_tuple(token_type_expand_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if token_type_expand_shape_tuple != (BATCH, SEQ_LEN):
        raise ValueError(f"unexpected token-type expand shape: {token_type_expand_shape!r}")
    if output_shape_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {output_shape!r}")

    device = word_embedding.device
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        token_type_source,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        token_type_expand_shape_tuple,
        output_shape_tuple,
    )


def _make_inductor_seed(device: torch.device) -> torch.Tensor:
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)
    return torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)


@oracle_impl(hardware="H100", shapes="(T([1, 512], i64, gen=Index(2)), T([1, 512], i64, gen=Index(512)), T([30522, 768], f32), T([32, 512], i64, gen=Index(30522)), T([2, 768], f32), T([512, 768], f32), T([768], f32), T([768], f32), S([32, 512]), S([16384, 768]))")
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
        raise RuntimeError("Triton is required for oracle_bert_embedding_layernorm_dropout.py")

    (
        token_type_source,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        _token_type_expand_shape,
        output_shape,
    ) = _validate_inputs(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=word_embedding.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=word_embedding.device,
        dtype=torch.float32,
    )
    seed = _make_inductor_seed(word_embedding.device)

    _bert_embedding_layernorm_dropout_kernel[(ROWS,)](
        token_type_source,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        seed,
        output,
        side,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    return torch.ops.aten.view.default(output, output_shape), side


def _normalize_outputs(outputs):
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _check_output_metadata(instance, inputs) -> bool:
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        actual = _normalize_outputs(oracle_forward(inputs))
        if actual and isinstance(actual[0], torch.Tensor) and actual[0].is_cuda:
            torch.cuda.synchronize()

    if len(actual) != len(eager):
        print(
            f"  SCOPE_MISMATCH metadata: oracle produces {len(actual)} outputs, "
            f"eager produces {len(eager)}"
        )
        return False

    all_ok = True
    for index, (expected, observed) in enumerate(zip(eager, actual)):
        ok = (
            isinstance(expected, torch.Tensor)
            and isinstance(observed, torch.Tensor)
            and expected.shape == observed.shape
            and expected.dtype == observed.dtype
            and expected.stride() == observed.stride()
        )
        print(
            f"  output {index} metadata: {'PASS' if ok else 'FAIL'} "
            f"(expected_shape={tuple(expected.shape)}, oracle_shape={tuple(observed.shape)}, "
            f"expected_stride={expected.stride()}, oracle_stride={observed.stride()}, "
            f"expected_dtype={expected.dtype}, oracle_dtype={observed.dtype})"
        )
        all_ok = all_ok and ok
    return all_ok


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


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
    if _repro_has_stochastic_ops():
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
        ok = _check_output_metadata(instance, inputs) and ok
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
