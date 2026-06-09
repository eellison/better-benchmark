"""
Oracle for amax_sum_9940b361e5b4

Gap diagnosis:
  Classification: NEW_PATTERN
  What oracle does differently: Computes the full Longformer sliding-window
    attention assembly, key/query masking, online row softmax, and final padded
    output layout directly in Triton.
  What Inductor change would fix: Add a Longformer sliding-window attention
    pattern lowering that fuses the structured band assembly with the softmax
    reduction epilogue and destination-layout scatter.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
SEQ_LEN = 4096
N_HEADS = 12
LOCAL_CHUNK = 256
CHUNKS = 16
BMM_CHUNKS = CHUNKS - 1
RNUMEL = 513
PADDED_RNUMEL = 770
FINAL_INNER = 769
OUT_M = N_HEADS * CHUNKS
OUT_T = LOCAL_CHUNK
OUT_D = 768
OUT_SHAPE = (OUT_M, OUT_T, OUT_D)
OUT_STRIDE = (LOCAL_CHUNK * PADDED_RNUMEL, FINAL_INNER, 1)
OUT_STORAGE_SIZE = (
    (OUT_M - 1) * OUT_STRIDE[0]
    + (OUT_T - 1) * OUT_STRIDE[1]
    + (OUT_D - 1) * OUT_STRIDE[2]
    + 1
)
ROWS = BATCH * SEQ_LEN * N_HEADS
BLOCK_N = 1024

CD_CONFIG = "coordinate_descent_tuning=True"
COMBO_CONFIG = (
    "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
    "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
    "triton.multi_kernel=3"
)
HISTORICAL_BEST_US = 340.86400270462036


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


if triton is not None:

    @triton.jit
    def _zero_kernel(out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < n_elements)

    @triton.jit
    def _longformer_online_softmax_kernel(
        query_mask_ptr,
        bmm_ptr,
        key_mask_ptr,
        out_ptr,
        BLOCK: tl.constexpr,
        R: tl.constexpr,
        SEQ: tl.constexpr,
        HEADS: tl.constexpr,
        CHUNK: tl.constexpr,
        CHUNKS_: tl.constexpr,
        BMM_CHUNKS_: tl.constexpr,
        OUT_D_: tl.constexpr,
        OUT_M_STRIDE: tl.constexpr,
        OUT_T_STRIDE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        valid_cols = cols < R

        head = row % HEADS
        seq = (row // HEADS) % SEQ
        chunk_id = seq // CHUNK
        pos = seq - chunk_id * CHUNK

        col_i32 = cols.to(tl.int32)
        key = seq + col_i32 - CHUNK
        valid_key = valid_cols & (key >= 0) & (key < SEQ)
        safe_key = tl.minimum(tl.maximum(key, 0), SEQ - 1)

        left_chunk = tl.minimum(safe_key // CHUNK, BMM_CHUNKS_ - 1)
        right_chunk = tl.minimum(chunk_id, BMM_CHUNKS_ - 1)
        source_chunk = tl.where(col_i32 < CHUNK, left_chunk, right_chunk)
        source_row = tl.minimum(tl.maximum(seq - source_chunk * CHUNK, 0), 511)
        source_col = tl.minimum(tl.maximum(safe_key - source_chunk * CHUNK, 0), 511)
        bmm_offsets = (
            (head * BMM_CHUNKS_ + source_chunk) * (512 * 512)
            + source_row * 512
            + source_col
        )

        local_scores = tl.load(bmm_ptr + bmm_offsets, mask=valid_key, other=-float("inf"))
        local_scores = local_scores.to(tl.float32)

        key_mask = tl.load(key_mask_ptr + safe_key, mask=valid_key, other=0.0)
        # In the captured graph the key-mask bias is added in fp16. For these
        # inputs that rounds any masked valid score to exactly -65504.
        scores = tl.where(key_mask != 0.0, -65504.0, local_scores)
        scores = tl.where(valid_key, scores, -float("inf"))

        row_max = tl.max(scores, axis=0)
        numer = tl.exp(scores - row_max)
        denom = tl.sum(numer, axis=0)
        values = numer / denom

        keep_row = tl.load(query_mask_ptr + seq) == 0
        out_m = head * CHUNKS_ + chunk_id
        out_d = pos + col_i32
        out_offsets = out_m * OUT_M_STRIDE + pos * OUT_T_STRIDE + out_d
        store_mask = valid_cols & (out_d < OUT_D_) & keep_row
        tl.store(out_ptr + out_offsets, values, mask=store_mask)


def _longformer_online_softmax(
    query_mask: torch.Tensor,
    bmm: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if bmm.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if query_mask.shape != (BATCH, SEQ_LEN) or query_mask.dtype != torch.bool:
        raise ValueError(f"unexpected query mask: shape={tuple(query_mask.shape)} dtype={query_mask.dtype}")
    if bmm.shape != (N_HEADS * BMM_CHUNKS, 512, 512) or bmm.dtype != torch.float16:
        raise ValueError(f"unexpected bmm: shape={tuple(bmm.shape)} dtype={bmm.dtype}")
    if key_mask.shape != (BATCH, SEQ_LEN) or key_mask.dtype != torch.float16:
        raise ValueError(f"unexpected key mask: shape={tuple(key_mask.shape)} dtype={key_mask.dtype}")

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=bmm.device, dtype=bmm.dtype)
    zero_block = 1024
    _zero_kernel[(triton.cdiv(OUT_STORAGE_SIZE, zero_block),)](
        out,
        OUT_STORAGE_SIZE,
        BLOCK=zero_block,
    )
    _longformer_online_softmax_kernel[(ROWS,)](
        query_mask,
        bmm,
        key_mask,
        out,
        BLOCK=BLOCK_N,
        R=RNUMEL,
        SEQ=SEQ_LEN,
        HEADS=N_HEADS,
        CHUNK=LOCAL_CHUNK,
        CHUNKS_=CHUNKS,
        BMM_CHUNKS_=BMM_CHUNKS,
        OUT_D_=OUT_D,
        OUT_M_STRIDE=OUT_STRIDE[0],
        OUT_T_STRIDE=OUT_STRIDE[1],
        num_warps=4,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([1, 4096], b8), T([180, 512, 512], f16), T([1, 4096], f16), S([12, 15, 512, 1, 512]), S([12, 15, 512, 512]), S([12, 15, 512, 513]), S([1, 12, 4096, 513]), S([1, 12, 4096, 513]), S([1, 256, 12, 257]), S([1, 12, 4096, 513]), S([12, 16, 256, 513]), S([1, 12, 4096, 513]), S([1, 12, 4096, 513]), S([1, 256, 12, 257]), S([1, 12, 4096, 513]), S([1, 4096, 1]), S([1, 8, 512, 1]), S([1, 4096, 1]), S([1, 8, 512, 1]), S([1, 15, 512, 512]), S([1, 15, 512, 513]), S([1, 1, 4096, 513]), S([1, 1, 4096, 513]), S([1, 256, 1, 257]), S([1, 1, 4096, 513]), S([1, 16, 256, 513]), S([1, 1, 4096, 513]), S([1, 1, 4096, 513]), S([1, 256, 1, 257]), S([1, 1, 4096, 513]), S([12, 16, 256, 513]), S([12, 16, -1]), S([12, 16, 256, 769]), S([192, 256, 768]))")
def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with a fused Longformer softmax oracle."""
    query_mask, bmm, key_mask = inputs[:3]
    return _longformer_online_softmax(query_mask, bmm, key_mask)


def _max_diffs(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(torch.nan_to_num(diff, nan=0.0, posinf=math.inf).max().item())
    rel = diff / expected.float().abs().clamp_min(1e-12)
    max_rel = float(torch.nan_to_num(rel, nan=0.0, posinf=math.inf).max().item())
    return max_abs, max_rel


@torch.no_grad()
def run_check(rtol: float, atol: float) -> bool:
    print(f"Checking {REPRO_ID}...")
    inputs = get_inputs()
    repro = get_repro_instance()
    expected = _as_tuple(repro(*inputs))
    torch.cuda.synchronize()
    actual = _as_tuple(oracle_forward(inputs))
    torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"  SCOPE_MISMATCH: expected {len(expected)} outputs, got {len(actual)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (act, exp) in enumerate(zip(actual, expected)):
        output_ok = True
        if act.shape != exp.shape:
            print(f"  output {idx}: FAIL shape expected={list(exp.shape)} actual={list(act.shape)}")
            output_ok = False
        if act.dtype != exp.dtype:
            print(f"  output {idx}: FAIL dtype expected={exp.dtype} actual={act.dtype}")
            output_ok = False
        if act.stride() != exp.stride():
            print(f"  output {idx}: FAIL stride expected={exp.stride()} actual={act.stride()}")
            output_ok = False
        close = torch.allclose(act.float(), exp.float(), rtol=rtol, atol=atol, equal_nan=True)
        max_abs, max_rel = _max_diffs(act, exp)
        output_ok = output_ok and bool(close)
        status = "PASS" if output_ok else "FAIL"
        print(
            f"  output {idx}: {status} "
            f"(shape={list(act.shape)} dtype={act.dtype} stride={act.stride()} "
            f"max_diff={max_abs:.2e} max_rel={max_rel:.2e})"
        )
        ok = ok and output_ok

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


@torch.no_grad()
def _time_cuda(fn, warmup: int, rep: int) -> tuple[float, float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[0], times[len(times) // 2], times


@torch.no_grad()
def run_bench(warmup: int, rep: int) -> dict[str, object]:
    print(f"Benchmarking {REPRO_ID}...")
    inputs = get_inputs()

    min_us, median_us, _ = _time_cuda(lambda: oracle_forward(inputs), warmup=warmup, rep=rep)
    true_floor = median_us < HISTORICAL_BEST_US
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(median_us, 3),
        "oracle_min_us": round(min_us, 3),
        "compile_us": None,
        "ratio": None,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "scope": "full_repro_forward",
        "classification": "NEW_PATTERN",
        "historical_best_compile_us": HISTORICAL_BEST_US,
        "true_floor_vs_historical_best": true_floor,
        "warmup": warmup,
        "rep": rep,
    }
    print(json.dumps(result, sort_keys=True))
    print(f"Oracle full-scope median: {median_us:.3f} us (min {min_us:.3f} us)")
    print(f"Historical best compile: {HISTORICAL_BEST_US:.3f} us")
    return result


def main() -> None:
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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
