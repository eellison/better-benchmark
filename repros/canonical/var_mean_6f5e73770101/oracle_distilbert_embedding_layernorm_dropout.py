"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DistilBERT embedding LayerNorm training scope in one fixed-hidden Triton row kernel, including word embedding lookup, sliced position embedding lookup, fp32 hidden-size-768 population var_mean with eps=1e-12, affine LayerNorm, Inductor-style dropout, and the final contiguous [32768, 768] view, whereas tuned Inductor already reaches the same practical embedding-gather, row-reduction, RNG/dropout, output-store, and launch envelope for this fixed shape; Inductor cannot materially improve this repro through a narrower scheduler-fusion or algebraic rewrite because the remaining work is dominated by mandatory embedding/affine reads, one row reduction, RNG generation, and output stores; the fix is BANDWIDTH_BOUND: record this as an at-floor embedding-layernorm-dropout case unless broader normalization, RNG, or launch-overhead improvements move the family. Exact stochastic equality with eager is skipped, so this measurement is not_true_floor."""
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
BATCH = 256
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
WORD_VOCAB = 30522
POSITION_SOURCE_LEN = 512
POSITION_VOCAB = 512
EPS = 1.0e-12
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 13
SEED_INDEX = 0
BLOCK_H = 1024
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)

if triton is not None:
    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _fast_embedding_layernorm_dropout_kernel(
        word_embedding_ptr,
        word_ids_ptr,
        position_ids_ptr,
        position_embedding_ptr,
        weight_ptr,
        bias_ptr,
        seeds_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < total_rows
        row_mask_1d = row_ids < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        seq = row_ids % seq_len

        word_id = tl.load(word_ids_ptr + row_ids, mask=row_mask_1d, other=0)[:, None]
        position_id = tl.load(position_ids_ptr + seq, mask=row_mask_1d, other=0)[:, None]
        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        x = tl.where(mask, word + position, 0.0)
        sum_x = tl.sum(x, axis=1)
        sum_x2 = tl.sum(x * x, axis=1)
        mean = sum_x / hidden
        variance = tl.maximum(sum_x2 / hidden - mean * mean, 0.0)
        invstd = tl.rsqrt(variance + 1.0e-12)
        centered = x - mean[:, None]

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        y = centered * invstd[:, None] * weight + bias

        offsets = rows * hidden + cols
        seed = tl.load(seeds_ptr)
        random = tl.rand(seed, offsets.to(tl.int32).to(tl.uint32))
        dropped = tl.where(random > dropout_p, y, 0.0) * dropout_scale
        tl.store(out_ptr + offsets, dropped, mask=mask)

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
    tuple[int, int],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    (
        word_embedding,
        word_ids,
        position_ids,
        position_embedding,
        weight,
        bias,
        output_shape,
    ) = inputs

    tensor_inputs = (
        word_embedding,
        word_ids,
        position_ids,
        position_embedding,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first six repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (WORD_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (1, POSITION_SOURCE_LEN),
        (POSITION_VOCAB, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    if word_ids.dtype != torch.int64 or position_ids.dtype != torch.int64:
        raise TypeError("word and position id inputs must be torch.int64")
    fp32_tensors = (word_embedding, position_embedding, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float32")

    device = word_embedding.device
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    output_shape_tuple = _shape_tuple(output_shape)
    if output_shape_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {output_shape!r}")

    return (
        word_embedding,
        word_ids,
        position_ids,
        position_embedding,
        weight,
        bias,
        output_shape_tuple,
    )


def _make_inductor_seed(device: torch.device) -> torch.Tensor:
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)
    return torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)


def _make_inductor_seeds(device: torch.device) -> torch.Tensor:
    seeds = torch.empty((SEED_COUNT,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(
        -9223372036854775808,
        9223372036854775807,
        [SEED_COUNT],
        out=seeds,
    )
    return seeds


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    (
        word_embedding,
        word_ids,
        position_ids,
        position_embedding,
        weight,
        bias,
        output_shape,
    ) = _validate_inputs(inputs)
    word = torch.ops.aten.embedding.default(word_embedding, word_ids, 0)
    pos_slice = torch.ops.aten.slice.Tensor(position_ids, 1, 0, SEQ_LEN)
    pos = torch.ops.aten.embedding.default(position_embedding, pos_slice)
    x = torch.ops.aten.add.Tensor(word, pos)
    var, mean = torch.ops.aten.var_mean.correction(x, [2], correction=0, keepdim=True)
    invstd = torch.ops.aten.rsqrt.default(var + EPS)
    y = (x - mean) * invstd
    y = y * weight + bias
    seed = _make_inductor_seed(word_embedding.device)
    random = torch.ops.prims.inductor_random.default((BATCH, SEQ_LEN, HIDDEN), seed, "rand")
    y = torch.ops.aten.mul.Tensor(torch.ops.aten.gt.Scalar(random, DROPOUT_P), y)
    y = torch.ops.aten.mul.Tensor(y, DROPOUT_SCALE)
    return torch.ops.aten.view.default(y, output_shape)


@oracle_impl(hardware="H100", shapes="(T([30522, 768], f32), T([256, 128], i64, gen=Index(30522)), T([1, 512], i64, gen=Index(512)), T([512, 768], f32), T([768], f32), T([768], f32), S([32768, 768]))")
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
    (
        word_embedding,
        word_ids,
        position_ids,
        position_embedding,
        weight,
        bias,
        output_shape,
    ) = _validate_inputs(inputs)
    if triton is None:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=word_embedding.device,
        dtype=torch.float32,
    )
    seeds = _make_inductor_seeds(word_embedding.device)
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _fast_embedding_layernorm_dropout_kernel[grid](
        word_embedding,
        word_ids,
        position_ids,
        position_embedding,
        weight,
        bias,
        seeds,
        output,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
    return torch.ops.aten.view.default(output, output_shape)


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
