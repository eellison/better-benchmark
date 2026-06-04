"""Gap diagnosis (classification: NEW_PATTERN): this oracle assembles the full Longformer sliding-window and global-mask attention scores, performs the row softmax/masking/Inductor-RNG dropout step, and writes the final strided [384, 768, 256] layout, whereas Inductor currently emits many separate slice/scatter/view/pad kernels around a generic softmax and layout pipeline; Inductor cannot do this today because the scheduler has no Longformer sliding-window attention pattern that fuses structured band assembly with the reduction epilogue and destination-layout scatter; the fix is NEW_PATTERN: add a Longformer sliding-window attention pattern lowering that recognizes the chunked band/global-mask assembly and generates a fused custom softmax/layout kernel."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_ID = "amax_sum_4c524f75213e"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ_LEN = 1024
N_HEADS = 12
RNUMEL = 513
LOCAL_CHUNK = 256
CHUNKS = 4
PADDED_RNUMEL = 770
FINAL_INNER = 769
OUT_M = BATCH * N_HEADS * CHUNKS
OUT_D = 768
OUT_T = 256
OUT_SHAPE = (OUT_M, OUT_D, OUT_T)
OUT_STRIDE = (LOCAL_CHUNK * PADDED_RNUMEL, 1, FINAL_INNER)
OUT_STORAGE_SIZE = (OUT_M - 1) * OUT_STRIDE[0] + (OUT_D - 1) + (OUT_T - 1) * OUT_STRIDE[2] + 1
ROWS = BATCH * SEQ_LEN * N_HEADS


@lru_cache(maxsize=1)
def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs():
    return _load_repro_module().make_inputs()


def get_repro_instance():
    return _load_repro_module().Repro().eval()


def _inductor_random_like_repro(dev: torch.device) -> torch.Tensor:
    seeds = torch.ops.prims.inductor_seeds.default(36, dev)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 0)
    return torch.ops.prims.inductor_random.default(
        [BATCH, SEQ_LEN, N_HEADS, RNUMEL],
        seed,
        "rand",
    )


@triton.jit
def _zero_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < n_elements)


@triton.jit
def _longformer_softmax_layout_kernel(
    bmm_ptr,
    arg7_ptr,
    arg8_ptr,
    random_ptr,
    out_ptr,
    BLOCK_N: tl.constexpr,
    R: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    CHUNK: tl.constexpr,
    OUT_D_: tl.constexpr,
    OUT_M_STRIDE: tl.constexpr,
    OUT_T_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid_cols = cols < R

    head = row % HEADS
    row_div_heads = row // HEADS
    seq = row_div_heads % SEQ
    batch = row_div_heads // SEQ
    bh = batch * HEADS + head

    key = seq + cols.to(tl.int32) - CHUNK
    valid_key = valid_cols & (key >= 0) & (key < SEQ)
    left_chunk = key // CHUNK
    right_chunk = tl.minimum(seq // CHUNK, 2)
    source_chunk = tl.where(cols < CHUNK, left_chunk, right_chunk)
    source_row = seq - source_chunk * CHUNK
    source_col = key - source_chunk * CHUNK
    bmm_offsets = (bh * 3 + source_chunk) * (512 * 512) + source_row * 512 + source_col

    local_scores = tl.load(bmm_ptr + bmm_offsets, mask=valid_key, other=-float("inf")).to(tl.float32)
    key_mask = tl.load(arg7_ptr + batch * SEQ + key, mask=valid_key, other=0.0).to(tl.float32)
    mask_bias = tl.where(key_mask != 0.0, -3.4028234663852886e38, 0.0)
    x = tl.where(valid_key, local_scores + mask_bias, -float("inf"))

    row_max = tl.max(x, axis=0)
    numer = tl.exp(x - row_max)
    denom = tl.sum(numer, axis=0)
    values = numer / denom

    keep_row = tl.load(arg8_ptr + batch * SEQ + seq) == 0
    keep_dropout = tl.load(random_ptr + row * R + cols, mask=valid_cols, other=0.0) > 1.0e-30

    chunk_id = seq // CHUNK
    pos = seq - chunk_id * CHUNK
    out_m = (batch * HEADS + head) * 4 + chunk_id

    padded_linear = pos * 770 + cols
    out_t = padded_linear // 769
    out_d = padded_linear - out_t * 769
    out_offsets = out_m * OUT_M_STRIDE + out_d + out_t * OUT_T_STRIDE
    store_mask = valid_cols & (out_d < OUT_D_) & keep_row & keep_dropout
    tl.store(out_ptr + out_offsets, values, mask=store_mask)


def _longformer_softmax_layout(
    bmm: torch.Tensor,
    arg7_1: torch.Tensor,
    arg8_1: torch.Tensor,
    random_values: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm.device,
        dtype=bmm.dtype,
    )
    zero_block = 1024
    _zero_kernel[(triton.cdiv(OUT_STORAGE_SIZE, zero_block),)](
        out,
        OUT_STORAGE_SIZE,
        BLOCK=zero_block,
    )
    _longformer_softmax_layout_kernel[(ROWS,)](
        bmm,
        arg7_1,
        arg8_1,
        random_values,
        out,
        BLOCK_N=1024,
        R=RNUMEL,
        SEQ=SEQ_LEN,
        HEADS=N_HEADS,
        CHUNK=LOCAL_CHUNK,
        OUT_D_=OUT_D,
        OUT_M_STRIDE=OUT_STRIDE[0],
        OUT_T_STRIDE=OUT_STRIDE[2],
        num_warps=4,
    )
    return out


def oracle_forward(inputs: tuple):
    """Compute the complete Repro.forward output for this Longformer row."""
    bmm, arg7_1, arg8_1 = inputs[:3]
    random_values = _inductor_random_like_repro(bmm.device)
    return _longformer_softmax_layout(bmm, arg7_1, arg8_1, random_values)


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _max_diffs(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual - expected).abs()
    max_abs = float(torch.nan_to_num(diff, nan=0.0, posinf=math.inf).max().item())
    rel = diff / expected.abs().clamp_min(1e-12)
    max_rel = float(torch.nan_to_num(rel, nan=0.0, posinf=math.inf).max().item())
    return max_abs, max_rel


@torch.no_grad()
def run_check(rtol: float, atol: float) -> bool:
    print(f"Checking {REPRO_ID}...")
    inputs = get_inputs()
    repro = get_repro_instance()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    expected = _as_tuple(repro(*inputs))
    torch.cuda.synchronize()
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    actual = _as_tuple(oracle_forward(inputs))
    torch.cuda.synchronize()

    ok = True
    if len(actual) != len(expected):
        print(f"  SCOPE_MISMATCH: expected {len(expected)} outputs, got {len(actual)}")
        return False

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
        close = torch.allclose(act, exp, rtol=rtol, atol=atol, equal_nan=True)
        max_abs, max_rel = _max_diffs(act, exp)
        output_ok = output_ok and bool(close)
        status = "PASS" if output_ok else "FAIL"
        print(
            f"  output {idx}: {status} "
            f"shape={list(act.shape)} dtype={act.dtype} stride={act.stride()} "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
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
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    return min(times), median, times


@torch.no_grad()
def run_bench(warmup: int, rep: int):
    print(f"Benchmarking {REPRO_ID}...")
    inputs = get_inputs()

    def bench_once():
        return oracle_forward(inputs)

    min_us, median_us, times = _time_cuda(bench_once, warmup=warmup, rep=rep)
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": median_us,
        "oracle_min_us": min_us,
        "warmup": warmup,
        "rep": rep,
        "scope": "full_repro_forward",
        "classification": "NEW_PATTERN",
    }
    print(json.dumps(result, sort_keys=True))
    print(f"Oracle full-scope median: {median_us:.3f} us (min {min_us:.3f} us)")
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Full-scope oracle for {REPRO_ID}")
    parser.add_argument("--check", action="store_true", help="check against Repro()(*make_inputs())")
    parser.add_argument("--bench", action="store_true", help="benchmark the full oracle path")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        return 1
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
