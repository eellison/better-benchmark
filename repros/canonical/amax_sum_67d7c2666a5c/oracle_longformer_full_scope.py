"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer sliding-window attention train scope returned by Repro.forward, including skewed bmm band assembly, first/last chunk triangular masking, arg7 key-mask bias, arg8 query zeroing, Inductor-RNG dropout, and the final padded strided [96,768,256] layout in one fused Triton row kernel, whereas Inductor currently lowers the decomposed pad/view/slice/scatter/mask/amax/sum/dropout/layout graph as separate generic assembly, reduction, random, and layout kernels over large intermediates; Inductor cannot do this today because its pattern library does not recognize Longformer diagonal chunk assembly as the producer of a stochastic attention-softmax epilogue with destination-layout scatter; the fix is NEW_PATTERN: add a Longformer sliding-window attention lowering that canonicalizes the structured band assembly, generated edge masks, key/query masks, dropout, and final layout into one fused schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 2
SEQ_LEN = 1024
N_HEADS = 12
BH = BATCH * N_HEADS
LOCAL_CHUNK = 256
CHUNKS = 4
BMM_CHUNKS = 3
RNUMEL = 513
PADDED_RNUMEL = 770
FINAL_INNER = 769
OUT_M = BH * CHUNKS
OUT_D = 768
OUT_T = 256
OUT_SHAPE = (OUT_M, OUT_D, OUT_T)
OUT_STRIDE = (LOCAL_CHUNK * PADDED_RNUMEL, 1, FINAL_INNER)
OUT_STORAGE_SIZE = (
    (OUT_M - 1) * OUT_STRIDE[0]
    + (OUT_D - 1) * OUT_STRIDE[1]
    + (OUT_T - 1) * OUT_STRIDE[2]
    + 1
)
ROWS = BATCH * SEQ_LEN * N_HEADS
BLOCK_N = 1024
N_SEEDS = 36
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
MASKED_BIAS = -3.4028234663852886e38

EXPECTED_SHAPE_PARAMS = (
    [24, 3, 512, 1, 512],
    [24, 3, 512, 512],
    [24, 3, 512, 513],
    [2, 256, 12, 257],
    [2, 12, 1024, 513],
    [24, 4, 256, 513],
    [2, 256, 12, 257],
    [2, 12, 1024, 513],
    [2, 1024, 1],
    [2, 1024, 1],
    [2, 2, 512, 1],
    [2, 2, 512, 1],
    [2, 3, 512, 512],
    [2, 3, 512, 513],
    [2, 256, 1, 257],
    [2, 1, 1024, 513],
    [2, 4, 256, 513],
    [2, 256, 1, 257],
    [2, 1, 1024, 513],
    [24, 4, 256, 513],
    [24, 4, -1],
    [24, 4, 256, 769],
    [96, 256, 768],
)


def _check_inputs(inputs):
    if len(inputs) != 26:
        raise ValueError(f"expected 26 repro inputs, got {len(inputs)}")
    bmm, arg7_1, arg8_1 = inputs[:3]
    if bmm.shape != (BH * BMM_CHUNKS, 512, 512) or bmm.dtype != torch.float32:
        raise ValueError(f"unexpected bmm: shape={tuple(bmm.shape)} dtype={bmm.dtype}")
    if arg7_1.shape != (BATCH, SEQ_LEN) or arg7_1.dtype != torch.float32:
        raise ValueError(f"unexpected arg7_1: shape={tuple(arg7_1.shape)} dtype={arg7_1.dtype}")
    if arg8_1.shape != (BATCH, SEQ_LEN) or arg8_1.dtype != torch.bool:
        raise ValueError(f"unexpected arg8_1: shape={tuple(arg8_1.shape)} dtype={arg8_1.dtype}")
    for idx, (actual, expected) in enumerate(zip(inputs[3:], EXPECTED_SHAPE_PARAMS), start=3):
        if list(actual) != expected:
            raise ValueError(f"unexpected shape param {idx}: {actual} != {expected}")


def _inductor_random_like_repro(device: torch.device) -> torch.Tensor:
    inductor_seeds = torch.ops.prims.inductor_seeds.default(N_SEEDS, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    return torch.ops.prims.inductor_random.default(
        [BATCH, SEQ_LEN, N_HEADS, RNUMEL],
        seed,
        "rand",
    )


if triton is not None:

    @triton.jit
    def _zero_kernel(out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < n_elements)

    @triton.jit
    def _load_skewed_bmm(
        bmm_ptr,
        bh,
        chunk,
        skew_row,
        skew_col,
        mask,
    ):
        linear = skew_row * 513 + skew_col
        src_row = linear // 512
        src_col = linear - src_row * 512
        valid = mask & (src_row < 512)
        safe_chunk = tl.where(valid, chunk, 0)
        safe_row = tl.where(valid, src_row, 0)
        safe_col = tl.where(valid, src_col, 0)
        offset = (bh * 3 + safe_chunk) * (512 * 512) + safe_row * 512 + safe_col
        return tl.load(bmm_ptr + offset, mask=valid, other=0.0).to(tl.float32)

    @triton.jit
    def _longformer_full_scope_kernel(
        bmm_ptr,
        arg7_ptr,
        arg8_ptr,
        random_ptr,
        out_ptr,
        BLOCK: tl.constexpr,
        R: tl.constexpr,
        SEQ: tl.constexpr,
        HEADS: tl.constexpr,
        CHUNK: tl.constexpr,
        CHUNKS_: tl.constexpr,
        OUT_D_: tl.constexpr,
        PADDED_R: tl.constexpr,
        FINAL_R: tl.constexpr,
        MASKED_BIAS_: tl.constexpr,
        DROPOUT_P_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        OUT_M_STRIDE: tl.constexpr,
        OUT_T_STRIDE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        valid_cols = cols < R

        head = row % HEADS
        row_div_heads = row // HEADS
        seq = row_div_heads % SEQ
        batch = row_div_heads // SEQ
        bh = batch * HEADS + head
        chunk_id = seq // CHUNK
        pos = seq - chunk_id * CHUNK

        col_i32 = cols.to(tl.int32)
        key = seq + col_i32 - CHUNK
        valid_key = valid_cols & (key >= 0) & (key < SEQ)

        from_right = col_i32 >= CHUNK
        source_chunk_right = tl.minimum(chunk_id, 2)
        skew_row_right = tl.where(chunk_id == 3, pos + CHUNK, pos)
        source_chunk_left = tl.where(chunk_id == 0, 0, chunk_id - 1)
        skew_row_left = tl.where(chunk_id == 0, pos - 1, pos + CHUNK - 1)
        source_chunk = tl.where(from_right, source_chunk_right, source_chunk_left)
        skew_row = tl.where(from_right, skew_row_right, skew_row_left)
        skew_col = tl.where(from_right, col_i32 - CHUNK, col_i32 + CHUNK + 1)

        local_score = _load_skewed_bmm(
            bmm_ptr,
            bh,
            source_chunk,
            skew_row,
            skew_col,
            valid_key,
        )

        safe_key = tl.minimum(tl.maximum(key, 0), SEQ - 1)
        key_is_masked = tl.load(arg7_ptr + batch * SEQ + safe_key, mask=valid_key, other=0.0) != 0.0
        key_bias = tl.where(key_is_masked, MASKED_BIAS_, 0.0)
        scores = tl.where(valid_key, local_score + key_bias, -float("inf"))

        row_max = tl.max(scores, axis=0)
        numer = tl.exp(scores - row_max)
        denom = tl.sum(numer, axis=0)
        values = numer / denom

        query_is_masked = tl.load(arg8_ptr + batch * SEQ + seq) != 0
        values = tl.where(query_is_masked, 0.0, values)

        random_values = tl.load(
            random_ptr + ((batch * SEQ + seq) * HEADS + head) * R + cols,
            mask=valid_cols,
            other=0.0,
        ).to(tl.float32)
        keep = random_values > DROPOUT_P_
        values = tl.where(keep, values * DROPOUT_SCALE_, 0.0)

        padded_linear = pos * PADDED_R + col_i32
        out_t = padded_linear // FINAL_R
        out_d = padded_linear - out_t * FINAL_R
        out_m = bh * CHUNKS_ + chunk_id
        out_offsets = out_m * OUT_M_STRIDE + out_d + out_t * OUT_T_STRIDE
        store_mask = valid_cols & (out_d < OUT_D_)
        tl.store(out_ptr + out_offsets, values, mask=store_mask)


def _launch_oracle(
    bmm: torch.Tensor,
    arg7_1: torch.Tensor,
    arg8_1: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if bmm.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    random_values = _inductor_random_like_repro(bmm.device)
    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=bmm.device, dtype=torch.float32)
    _zero_kernel[(triton.cdiv(OUT_STORAGE_SIZE, BLOCK_N),)](
        out,
        OUT_STORAGE_SIZE,
        BLOCK=BLOCK_N,
    )
    _longformer_full_scope_kernel[(ROWS,)](
        bmm,
        arg7_1,
        arg8_1,
        random_values,
        out,
        BLOCK=BLOCK_N,
        R=RNUMEL,
        SEQ=SEQ_LEN,
        HEADS=N_HEADS,
        CHUNK=LOCAL_CHUNK,
        CHUNKS_=CHUNKS,
        OUT_D_=OUT_D,
        PADDED_R=PADDED_RNUMEL,
        FINAL_R=FINAL_INNER,
        MASKED_BIAS_=MASKED_BIAS,
        DROPOUT_P_=DROPOUT_P,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        OUT_M_STRIDE=OUT_STRIDE[0],
        OUT_T_STRIDE=OUT_STRIDE[2],
        num_warps=4,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([72, 512, 512], f32), T([2, 1024], f32), T([2, 1024], b8), S([24, 3, 512, 1, 512]), S([24, 3, 512, 512]), S([24, 3, 512, 513]), S([2, 256, 12, 257]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 256, 12, 257]), S([2, 12, 1024, 513]), S([2, 1024, 1]), S([2, 1024, 1]), S([2, 2, 512, 1]), S([2, 2, 512, 1]), S([2, 3, 512, 512]), S([2, 3, 512, 513]), S([2, 256, 1, 257]), S([2, 1, 1024, 513]), S([2, 4, 256, 513]), S([2, 256, 1, 257]), S([2, 1, 1024, 513]), S([24, 4, 256, 513]), S([24, 4, -1]), S([24, 4, 256, 769]), S([96, 256, 768]))")
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
    _check_inputs(inputs)
    bmm, arg7_1, arg8_1 = inputs[:3]
    return _launch_oracle(bmm, arg7_1, arg8_1)


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
