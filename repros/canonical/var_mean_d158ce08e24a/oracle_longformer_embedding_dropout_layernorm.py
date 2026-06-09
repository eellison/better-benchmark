"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer embedding-layernorm-dropout Repro.forward scope in one fixed-hidden Triton row kernel, including mask-derived position ids, word/position/global embedding gathers, fp32 hidden-size-768 var_mean normalization, affine, Inductor-seeded dropout, and the rsqrt/768 side output, whereas Inductor currently lowers the same graph through separate generic embedding gathers, normalization-template reduction, stochastic dropout pointwise work, and side-output scheduling; Inductor cannot do this today because its pattern matcher and normalization scheduler do not canonicalize dynamic Longformer position-id construction plus multiple gathered embedding producers, fixed-K layernorm, stochastic epilogue, and live inverse-std output into one template; the fix is NEW_PATTERN: add a Longformer embedding-layernorm-dropout lowering that folds position-id construction, indexed embedding loads, fixed-K row reduction, affine, RNG dropout, and side-output storage into specialized codegen."""
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

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 2
SEQ_LEN = 1024
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
WORD_VOCAB = 50265
POSITION_VOCAB = 4098
GLOBAL_VOCAB = 1
EPS = 1.0e-5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 1
SEED_INDEX = 0
BLOCK_H = 1024
STOCHASTIC_OUTPUTS = (0,)
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _longformer_embedding_dropout_layernorm_kernel(
        cumsum_ptr,
        position_mask_ptr,
        word_embedding_ptr,
        word_ids_ptr,
        position_embedding_ptr,
        global_embedding_ptr,
        weight_ptr,
        bias_ptr,
        seed_ptr,
        dropped_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        cumsum_i32 = tl.load(cumsum_ptr + row).to(tl.int32)
        position_mask = tl.load(position_mask_ptr + row)
        position_id = (cumsum_i32 * position_mask).to(tl.int64) + 1
        word_id = tl.load(word_ids_ptr + row)

        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        global_token = tl.load(
            global_embedding_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        x = tl.where(mask, word + position + global_token, 0.0)
        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normalized = centered * invstd * weight + bias

        flat_offsets = row * hidden + cols
        seed = tl.load(seed_ptr)
        random = tl.rand(seed, flat_offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, normalized, 0.0) * dropout_scale

        tl.store(dropped_ptr + flat_offsets, dropped, mask=mask)
        tl.store(invstd_div_ptr + row, invstd / hidden)


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
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        cumsum,
        position_mask,
        word_embedding,
        word_ids,
        position_embedding,
        global_embedding,
        weight,
        bias,
    ) = inputs

    tensor_inputs = (
        cumsum,
        position_mask,
        word_embedding,
        word_ids,
        position_embedding,
        global_embedding,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("all repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (BATCH, SEQ_LEN),
        (BATCH, SEQ_LEN),
        (WORD_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (POSITION_VOCAB, HIDDEN),
        (GLOBAL_VOCAB, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    if cumsum.dtype != torch.int64 or word_ids.dtype != torch.int64:
        raise TypeError("cumsum and word ids must be torch.int64")
    if position_mask.dtype != torch.int32:
        raise TypeError(f"position mask must be torch.int32, got {position_mask.dtype}")
    fp32_tensors = (word_embedding, position_embedding, global_embedding, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float32")

    return (
        cumsum,
        position_mask,
        word_embedding,
        word_ids,
        position_embedding,
        global_embedding,
        weight,
        bias,
    )


def _make_inductor_seed(device: torch.device) -> torch.Tensor:
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)
    return torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)


@oracle_impl(hardware="H100", shapes="(T([2, 1024], i64, gen=Index(4097)), T([2, 1024], i32, gen=Index(2)), T([50265, 768], f32), T([2, 1024], i64, gen=Index(50265)), T([4098, 768], f32), T([1, 768], f32), T([768], f32), T([768], f32))")
def oracle_forward(inputs):
    """Run the complete Repro.forward scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_longformer_embedding_dropout_layernorm.py")

    (
        cumsum,
        position_mask,
        word_embedding,
        word_ids,
        position_embedding,
        global_embedding,
        weight,
        bias,
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

    _longformer_embedding_dropout_layernorm_kernel[(ROWS,)](
        cumsum,
        position_mask,
        word_embedding,
        word_ids,
        position_embedding,
        global_embedding,
        weight,
        bias,
        seed,
        dropped,
        invstd_div,
        hidden=HIDDEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        block_h=BLOCK_H,
        num_warps=2,
        num_stages=3,
    )
    return (dropped, invstd_div)


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


def _check_longformer_oracle(
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

        metadata = (
            f"shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={list(expected.stride())}"
        )
        if not _tensor_metadata_matches(expected, actual):
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
                        help="Disable skipping the stochastic dropout output")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
        ok = _check_longformer_oracle(
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
