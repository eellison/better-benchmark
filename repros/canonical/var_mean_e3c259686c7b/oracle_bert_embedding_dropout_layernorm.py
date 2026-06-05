"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BERT word/token-type/position embedding assembly, fp32 hidden-size-768 layernorm, affine, Inductor-style dropout, final [2048,768] view, and rsqrt/768 side output in one fixed-hidden Triton row kernel, whereas Inductor currently lowers the graph through generic embedding gathers, norm-template reduction, RNG dropout, pointwise, and view scheduling; Inductor cannot do this today because norm-template canonicalization does not recognize gathered embedding producers plus stochastic dropout and a live inverse-std side output as one BERT embedding-layernorm-dropout codegen pattern; the fix is NEW_PATTERN: add a BERT embedding-layernorm-dropout template that folds the gather producers, fixed-K row reduction, affine dropout epilogue, final view, and side-output store into specialized Inductor codegen."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)

ROWS = 4 * 512
BATCH = 4
SEQ_LEN = 512
HIDDEN = 768
WORD_VOCAB = 30522
TOKEN_TYPES = 2
EPS = 1.0e-12
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 37
SEED_INDEX = 0
SEED_LOW = -9223372036854775808
SEED_HIGH = 9223372036854775807
BLOCK_H = 1024
STOCHASTIC_OUTPUTS = (0,)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        token_type_ids_ptr,
        position_ids_ptr,
        word_embedding_ptr,
        word_ids_ptr,
        token_type_embedding_ptr,
        position_embedding_ptr,
        weight_ptr,
        bias_ptr,
        seeds_ptr,
        dropped_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        seed_index: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        seq = row % seq_len
        word_id = tl.load(word_ids_ptr + row)
        position_id = tl.load(position_ids_ptr + seq)
        token_type_id = tl.load(token_type_ids_ptr + position_id)

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

        flat_offsets = row * hidden + cols
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, flat_offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, y, 0.0) * dropout_scale

        tl.store(dropped_ptr + flat_offsets, dropped, mask=mask)
        tl.store(invstd_div_ptr + row, invstd / hidden)


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
        token_type_ids,
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
        token_type_ids,
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
        (SEQ_LEN, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    index_tensors = (token_type_ids, position_ids, word_ids)
    if not all(value.dtype == torch.int64 for value in index_tensors):
        raise TypeError("token type ids, position ids, and word ids must be torch.int64")
    fp32_tensors = (word_embedding, token_type_embedding, position_embedding, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float32")

    token_type_expand_shape_tuple = _shape_tuple(token_type_expand_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if token_type_expand_shape_tuple != (BATCH, SEQ_LEN):
        raise ValueError(f"unexpected token-type expand shape: {token_type_expand_shape!r}")
    if output_shape_tuple != (ROWS, HIDDEN):
        raise ValueError(f"unexpected output view shape: {output_shape!r}")

    return (
        token_type_ids,
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


def _make_inductor_seeds(device: torch.device) -> torch.Tensor:
    """Match compiled Inductor's lowering of prims.inductor_seeds.default(37)."""
    seeds = torch.empty_strided((SEED_COUNT,), (1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(SEED_LOW, SEED_HIGH, [SEED_COUNT], out=seeds)
    return seeds


def _normalize_outputs(outputs: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, (list, tuple)):
        return tuple(outputs)
    raise TypeError(f"expected tensor output or tuple/list of tensors, got {type(outputs).__name__}")


def check_oracle_metadata_and_values(
    inputs: list[Any] | tuple[Any, ...],
    instance: torch.nn.Module,
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool,
) -> bool:
    """Check full output metadata while skipping only stochastic value equality."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)

    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    stochastic = set(STOCHASTIC_OUTPUTS) if skip_stochastic else set()
    for i, (eager_value, oracle_value) in enumerate(zip(eager_list, oracle_list)):
        metadata_ok = True
        if not isinstance(eager_value, torch.Tensor) or not isinstance(oracle_value, torch.Tensor):
            print(f"  output {i}: SCOPE_MISMATCH non-tensor output")
            all_pass = False
            continue
        if eager_value.shape != oracle_value.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(oracle_value.shape)} "
                f"eager={list(eager_value.shape)}"
            )
            metadata_ok = False
        if eager_value.dtype != oracle_value.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={oracle_value.dtype} "
                f"eager={eager_value.dtype}"
            )
            metadata_ok = False
        if eager_value.stride() != oracle_value.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={list(oracle_value.stride())} "
                f"eager={list(eager_value.stride())}"
            )
            metadata_ok = False
        if eager_value.device != oracle_value.device:
            print(
                f"  output {i}: SCOPE_MISMATCH device oracle={oracle_value.device} "
                f"eager={eager_value.device}"
            )
            metadata_ok = False
        if not metadata_ok:
            all_pass = False
            continue

        if i in stochastic:
            print(
                f"  output {i}: SKIP value (stochastic, metadata PASS: "
                f"shape={list(eager_value.shape)} dtype={eager_value.dtype} "
                f"stride={list(eager_value.stride())})"
            )
            continue

        if not eager_value.is_floating_point():
            ok = torch.equal(eager_value, oracle_value)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={eager_value.dtype})")
            if not ok:
                all_pass = False
            continue

        eager_f32 = eager_value.float()
        oracle_f32 = oracle_value.float()
        max_diff = (eager_f32 - oracle_f32).abs().max().item()
        ok = torch.allclose(eager_f32, oracle_f32, atol=atol, rtol=rtol)
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(eager_value.shape)} dtype={eager_value.dtype} max_diff={max_diff:.2e})"
        )
        if not ok:
            all_pass = False

    return all_pass


def oracle_forward(inputs):
    """Run the complete Repro.forward scope.

    This includes the token-type expand/gather, word/token-type/position
    embedding gathers, fp32 layernorm with correction=0 and eps=1e-12, affine,
    explicit Inductor rand dropout at p=0.1 with scale 1/0.9, final view, and
    the rsqrt/768 side output.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bert_embedding_dropout_layernorm.py")

    (
        token_type_ids,
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

    dropped = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=word_embedding.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=word_embedding.device,
        dtype=torch.float32,
    )
    seeds = _make_inductor_seeds(word_embedding.device)

    _embedding_layernorm_kernel[(ROWS,)](
        token_type_ids,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        seeds,
        dropped,
        invstd_div,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        seed_index=SEED_INDEX,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    return (torch.ops.aten.view.default(dropped, output_shape), invstd_div)


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
    if has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text():
        if args.no_skip_stochastic:
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; --no-skip-stochastic "
                "requested, so dropout values will be compared"
            )
        else:
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; output 0 metadata is "
                "checked and only output 0 values are skipped"
            )

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle_metadata_and_values(
            inputs,
            instance,
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
