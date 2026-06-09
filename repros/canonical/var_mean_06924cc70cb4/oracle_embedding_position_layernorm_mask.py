"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DistilGPT2 token-plus-generated-position embedding LayerNorm scope in one shape-specialized Triton row kernel that keeps the full hidden-size-768 embedding tile live through Welford population var_mean, eps=1e-5 libdevice rsqrt, affine epilogue, final contiguous `[16384, 768]` view, and the adjacent-position bool side output as an all-false store, whereas Inductor emits a fused embedding/LayerNorm Welford kernel that reloads the token and position embeddings for the epilogue and separately materializes generated position ids through cat/slice/sub/ne kernels for the provably all-false mask; Inductor cannot do this today because the generic row-reduction scheduler does not retain a full fixed-width producer tile across Welford reductions or sink algebraically constant sibling side-output stores into the normalization epilogue; the fix is SCHEDULER_FUSION: teach the fixed-hidden normalization schedule to keep safe producer tiles resident through affine consumers and attach constant side-output stores when the sibling mask pattern has been simplified."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


BATCH = 32
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
TOKEN_VOCAB = 50257
POSITION_ROWS = 1024
EPS = 1.0e-5
BLOCK_H = 1024
XBLOCK = 2
DEFAULT_NUM_WARPS = 4
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice

    @triton.jit
    def _embedding_position_layernorm_mask_kernel(
        token_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        mask_out_ptr,
        n_rows: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        xblock: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * xblock + tl.arange(0, xblock)
        row_mask = row_offsets < n_rows
        cols = tl.arange(0, block_h)
        cols_2d = cols[None, :]
        row_2d = row_offsets[:, None]
        mask = row_mask[:, None] & (cols_2d < hidden)

        token_id = tl.load(token_ids_ptr + row_offsets, mask=row_mask, other=0)[:, None]
        position_id = (row_offsets % seq_len)[:, None]

        position = tl.load(
            position_table_ptr + position_id * hidden + cols_2d,
            mask=mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        token = tl.load(
            token_table_ptr + token_id * hidden + cols_2d,
            mask=mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        values = tl.where(mask, token + position, 0.0)
        # Hidden=768 fits one block; use Inductor's generated
        # welford_reduce -> welford sequence for correction=0 var_mean.
        mean_acc = tl.zeros([xblock, block_h], tl.float32)
        m2_acc = tl.zeros([xblock, block_h], tl.float32)
        weight_acc = tl.zeros([xblock, block_h], tl.float32)
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            values, mean_acc, m2_acc, weight_acc, True
        )
        mean_acc = tl.where(mask, mean_next, mean_acc)
        m2_acc = tl.where(mask, m2_next, m2_acc)
        weight_acc = tl.where(mask, weight_next, weight_acc)
        mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)

        weight = tl.load(
            weight_ptr + cols_2d,
            mask=cols_2d < hidden,
            eviction_policy="evict_last",
            other=0.0,
        )
        bias = tl.load(
            bias_ptr + cols_2d,
            mask=cols_2d < hidden,
            eviction_policy="evict_last",
            other=0.0,
        )
        centered = values - mean[:, None]
        denom = tl.full([xblock], 768.0, tl.float32)
        eps_value = tl.full([xblock], eps, tl.float32)
        variance = m2 / denom
        invstd = libdevice.rsqrt(variance + eps_value)[:, None]
        y = centered * invstd
        y = y * weight
        y = y + bias
        tl.store(out_ptr + row_2d * hidden + cols_2d, y, mask=mask)

        tl.store(mask_out_ptr + row_offsets, False, mask=row_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _matches_shape(actual: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    return len(actual) == len(expected) and all(
        got == want or got == -1 or want == -1
        for got, want in zip(actual, expected)
    )


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    token_table, token_ids, position_table, weight, bias, view_shape, expand_shape = inputs
    tensor_inputs = (token_table, token_ids, position_table, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (TOKEN_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (POSITION_ROWS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if token_table.dtype != torch.float32 or position_table.dtype != torch.float32:
        raise TypeError("embedding tables must be torch.float32")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("LayerNorm weight and bias must be torch.float32")
    if token_ids.dtype != torch.int64:
        raise TypeError(f"token ids must be torch.int64, got {token_ids.dtype}")
    if (
        token_ids.device != token_table.device
        or position_table.device != token_table.device
        or weight.device != token_table.device
        or bias.device != token_table.device
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    view_shape_tuple = _shape_tuple(view_shape)
    expand_shape_tuple = _shape_tuple(expand_shape)
    if not _matches_shape(view_shape_tuple, (ROWS, HIDDEN)):
        raise ValueError(f"unexpected output view shape parameter: {view_shape!r}")
    if not _matches_shape(expand_shape_tuple, (BATCH, SEQ_LEN)):
        raise ValueError(f"unexpected expand shape parameter: {expand_shape!r}")

    return token_table, token_ids, position_table, weight, bias, view_shape_tuple


def oracle_embedding_position_layernorm_mask(
    token_table: torch.Tensor,
    token_ids: torch.Tensor,
    position_table: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    view_shape: tuple[int, ...],
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the complete deterministic Repro.forward output contract."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_position_layernorm_mask.py")

    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    mask = torch.empty_strided(
        (BATCH, SEQ_LEN),
        (SEQ_LEN, 1),
        device=token_table.device,
        dtype=torch.bool,
    )
    _embedding_position_layernorm_mask_kernel[(triton.cdiv(ROWS, XBLOCK),)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out,
        mask,
        n_rows=ROWS,
        seq_len=SEQ_LEN,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        xblock=XBLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out.view(view_shape), mask


@oracle_impl(hardware="H100", shapes="(T([50257, 768], f32), T([32, 512], i64, gen=Index(50257)), T([1024, 768], f32), T([768], f32), T([768], f32), S([-1, 768]), S([32, -1]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope: embedding, LayerNorm, view, and mask."""
    token_table, token_ids, position_table, weight, bias, view_shape = _validate_inputs(inputs)
    return oracle_embedding_position_layernorm_mask(
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        view_shape,
    )


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
