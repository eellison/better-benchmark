"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete RoBERTa embedding-layernorm-dropout Repro.forward scope in one fixed-hidden Triton row kernel, including cumsum/mask position-id construction, gathered segment ids, word/segment/position embedding gathers, fp32 layernorm with eps=1e-5, affine, generated-seed dropout, final [2048,768] view, and deterministic rsqrt/768 side output while --check skips only the stochastic dropout value comparison so true_floor=no, whereas Inductor currently lowers the same graph through generic embedding gather, norm-template reduction, RNG/dropout, pointwise, and view scheduling; Inductor cannot do this today because its pattern matcher and normalization scheduler do not canonicalize dynamic RoBERTa position-id construction plus gathered embedding producers, fixed-K layernorm, stochastic dropout, and a live inverse-std side output into one embedding-layernorm-dropout template; the fix is NEW_PATTERN: add a RoBERTa embedding-layernorm-dropout lowering that folds dynamic position-id construction, indexed embedding loads, fixed-K row reduction, dropout, final view, and side-output storage into specialized codegen."""
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


ROWS = 4 * 512
BATCH = 4
SEQ_LEN = 512
POSITION_VOCAB = 514
HIDDEN = 768
WORD_VOCAB = 250002
SEGMENT_VOCAB = 1
EPS = 1.0e-5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 37
SEED_INDEX = 0
BLOCK_H = 1024
EXACT_STOCHASTIC_EQUALITY = False
STOCHASTIC_OUTPUTS = (0,)

if triton is not None:

    @triton.jit
    def _roberta_embedding_layernorm_dropout_kernel(
        cumsum_ptr,
        position_mask_ptr,
        segment_id_source_ptr,
        word_embedding_ptr,
        word_ids_ptr,
        segment_embedding_ptr,
        position_embedding_ptr,
        weight_ptr,
        bias_ptr,
        seed_ptr,
        dropped_ptr,
        invstd_div_ptr,
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

        cumsum_i32 = tl.load(cumsum_ptr + row).to(tl.int32)
        position_mask_i32 = tl.load(position_mask_ptr + row)
        position_id = (cumsum_i32 * position_mask_i32).to(tl.int64) + 1
        segment_id = tl.load(segment_id_source_ptr + position_id)
        word_id = tl.load(word_ids_ptr + row)

        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        segment = tl.load(
            segment_embedding_ptr + segment_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        x = tl.where(mask, word + segment + position, 0.0)
        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        flat_offsets = row * hidden + cols
        seed = tl.load(seed_ptr)
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
    torch.Tensor,
    tuple[int, int],
]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        cumsum,
        position_mask,
        segment_id_source,
        word_embedding,
        word_ids,
        segment_embedding,
        position_embedding,
        weight,
        bias,
        expand_shape,
        segment_expand_shape,
        output_shape,
    ) = inputs

    tensor_inputs = (
        cumsum,
        position_mask,
        segment_id_source,
        word_embedding,
        word_ids,
        segment_embedding,
        position_embedding,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first nine repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (BATCH, SEQ_LEN),
        (BATCH, SEQ_LEN),
        (1, POSITION_VOCAB),
        (WORD_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (SEGMENT_VOCAB, HIDDEN),
        (POSITION_VOCAB, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    if cumsum.dtype != torch.int64:
        raise TypeError(f"cumsum must be torch.int64, got {cumsum.dtype}")
    if position_mask.dtype != torch.int32:
        raise TypeError(f"position mask must be torch.int32, got {position_mask.dtype}")
    if segment_id_source.dtype != torch.int64 or word_ids.dtype != torch.int64:
        raise TypeError("segment id source and word ids must be torch.int64")
    fp32_tensors = (word_embedding, segment_embedding, position_embedding, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float32")

    expand_shape_tuple = _shape_tuple(expand_shape)
    segment_expand_shape_tuple = _shape_tuple(segment_expand_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if expand_shape_tuple != (BATCH, -1):
        raise ValueError(f"unexpected source expand shape: {expand_shape!r}")
    if segment_expand_shape_tuple != (BATCH, SEQ_LEN):
        raise ValueError(f"unexpected segment expand shape: {segment_expand_shape!r}")
    if output_shape_tuple != (ROWS, HIDDEN):
        raise ValueError(f"unexpected output view shape: {output_shape!r}")

    return (
        cumsum,
        position_mask,
        segment_id_source,
        word_embedding,
        word_ids,
        segment_embedding,
        position_embedding,
        weight,
        bias,
        output_shape_tuple,
    )


def _make_inductor_seed(device: torch.device) -> torch.Tensor:
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)
    return torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)


@oracle_impl(hardware="H100", shapes="(T([4, 512], i64, gen=Index(513)), T([4, 512], i32, gen=Index(2)), T([1, 514], i64, gen=Index(1)), T([250002, 768], f32), T([4, 512], i64, gen=Index(250002)), T([1, 768], f32), T([514, 768], f32), T([768], f32), T([768], f32), S([4, -1]), S([4, 512]), S([2048, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_roberta_embedding_dropout_layernorm.py")

    (
        cumsum,
        position_mask,
        segment_id_source,
        word_embedding,
        word_ids,
        segment_embedding,
        position_embedding,
        weight,
        bias,
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
    seed = _make_inductor_seed(word_embedding.device)

    _roberta_embedding_layernorm_dropout_kernel[(ROWS,)](
        cumsum,
        position_mask,
        segment_id_source,
        word_embedding,
        word_ids,
        segment_embedding,
        position_embedding,
        weight,
        bias,
        seed,
        dropped,
        invstd_div,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        block_h=BLOCK_H,
        num_warps=2,
        num_stages=3,
    )
    return (torch.ops.aten.view.default(dropped, output_shape), invstd_div)


def _normalize_outputs(outputs):
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _tensor_metadata_matches(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return (
        expected.shape == actual.shape
        and expected.dtype == actual.dtype
        and expected.stride() == actual.stride()
    )


def _check_roberta_oracle(
    instance,
    inputs,
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool,
) -> bool:
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        oracle_out = _normalize_outputs(oracle_forward(inputs))

    if len(oracle_out) != len(eager):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_out)} outputs, "
            f"eager produces {len(eager)}"
        )
        return False

    all_pass = True
    for index, (expected, actual) in enumerate(zip(eager, oracle_out)):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            ok = expected == actual
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        metadata_ok = _tensor_metadata_matches(expected, actual)
        metadata = (
            f"shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={list(expected.stride())}"
        )
        if not metadata_ok:
            print(
                f"  output {index}: SCOPE_MISMATCH metadata "
                f"oracle_shape={list(actual.shape)} eager_shape={list(expected.shape)} "
                f"oracle_dtype={actual.dtype} eager_dtype={expected.dtype} "
                f"oracle_stride={list(actual.stride())} eager_stride={list(expected.stride())}"
            )
            all_pass = False
            continue

        if skip_stochastic and index in STOCHASTIC_OUTPUTS:
            print(f"  output {index}: SKIP values (stochastic dropout; metadata PASS {metadata})")
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (exact, {metadata})")
            all_pass = all_pass and bool(ok)
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        max_diff = (expected_f32 - actual_f32).abs().max().item()
        ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
        print(
            f"  output {index}: {'PASS' if ok else 'FAIL'} "
            f"({metadata} max_diff={max_diff:.2e})"
        )
        all_pass = all_pass and bool(ok)

    return all_pass


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
                "requested, so dropout-dependent values will be compared"
            )
        else:
            skipped = ", ".join(str(index) for index in STOCHASTIC_OUTPUTS)
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; only value comparison "
                f"for output {skipped} is skipped"
            )

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_roberta_oracle(
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
