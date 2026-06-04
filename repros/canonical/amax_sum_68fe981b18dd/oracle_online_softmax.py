"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer inference sliding-window attention path by assembling the structured local scores and key mask directly from `bmm_22`/`arg7_1`, applying the row padding mask, performing the full row softmax, and writing the final strided `[384, 256, 768]` output layout, whereas Inductor currently emits many separate slice/scatter/pad/view kernels around a generic softmax and final layout pipeline; Inductor cannot do this today because its scheduler/codegen lacks a Longformer sliding-window attention pattern that fuses structured band assembly, mask insertion, row softmax, and destination-layout scatter into one custom lowering; the fix is NEW_PATTERN: add a Longformer sliding-window attention lowering that recognizes the chunked band/global-mask assembly and generates a fused softmax/layout kernel for the required output view."""
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

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None


REPRO_ID = "amax_sum_68fe981b18dd"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

sys.path.insert(0, str(REPO_ROOT))

BATCH = 8
SEQ_LEN = 1024
N_HEADS = 12
RNUMEL = 513
LOCAL_CHUNK = 256
CHUNKS = 4
PADDED_RNUMEL = 770
FINAL_INNER = 769
OUT_M = BATCH * N_HEADS * CHUNKS
OUT_T = 256
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


@lru_cache(maxsize=1)
def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs() -> tuple[object, ...]:
    return _load_repro_module().make_inputs()


def get_repro_instance() -> torch.nn.Module:
    return _load_repro_module().Repro().eval()


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


if triton is not None:

    @triton.jit
    def _zero_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < n_elements)

    @triton.jit
    def _longformer_softmax_layout_kernel(
        query_mask_ptr,
        bmm_ptr,
        key_mask_ptr,
        out_ptr,
        BLOCK_N: tl.constexpr,
        R: tl.constexpr,
        SEQ: tl.constexpr,
        HEADS: tl.constexpr,
        CHUNK: tl.constexpr,
        PADDED_R: tl.constexpr,
        FINAL_INNER_: tl.constexpr,
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
        safe_key = tl.minimum(tl.maximum(key, 0), SEQ - 1)
        left_chunk = safe_key // CHUNK
        right_chunk = tl.minimum(seq // CHUNK, 2)
        source_chunk = tl.where(cols < CHUNK, left_chunk, right_chunk)
        source_row = seq - source_chunk * CHUNK
        source_col = safe_key - source_chunk * CHUNK
        source_row = tl.minimum(tl.maximum(source_row, 0), 511)
        source_col = tl.minimum(tl.maximum(source_col, 0), 511)
        bmm_offsets = (bh * 3 + source_chunk) * (512 * 512) + source_row * 512 + source_col

        local_scores = tl.load(bmm_ptr + bmm_offsets, mask=valid_key, other=-float("inf")).to(tl.float32)
        key_mask = tl.load(key_mask_ptr + batch * SEQ + safe_key, mask=valid_key, other=0.0).to(tl.float32)
        mask_bias = tl.where(key_mask != 0.0, -3.4028234663852886e38, 0.0)
        scores = tl.where(valid_key, local_scores + mask_bias, -float("inf"))

        row_max = tl.max(scores, axis=0)
        numer = tl.exp(scores - row_max)
        denom = tl.sum(numer, axis=0)
        values = numer / denom

        keep_row = tl.load(query_mask_ptr + batch * SEQ + seq) == 0
        chunk_id = seq // CHUNK
        pos = seq - chunk_id * CHUNK
        out_m = (batch * HEADS + head) * 4 + chunk_id

        safe_cols = tl.minimum(cols, R - 1)
        padded_linear = pos * PADDED_R + safe_cols
        out_t = padded_linear // FINAL_INNER_
        out_d = padded_linear - out_t * FINAL_INNER_
        out_offsets = out_m * OUT_M_STRIDE + out_t * OUT_T_STRIDE + out_d
        store_mask = valid_cols & (out_d < OUT_D_) & keep_row
        tl.store(out_ptr + out_offsets, values, mask=store_mask)


def _longformer_softmax_layout(
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
    if bmm.shape != (BATCH * N_HEADS * 3, 512, 512) or bmm.dtype != torch.float32:
        raise ValueError(f"unexpected bmm: shape={tuple(bmm.shape)} dtype={bmm.dtype}")
    if key_mask.shape != (BATCH, SEQ_LEN) or key_mask.dtype != torch.float32:
        raise ValueError(f"unexpected key mask: shape={tuple(key_mask.shape)} dtype={key_mask.dtype}")

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=bmm.device, dtype=torch.float32)
    zero_block = 1024
    _zero_kernel[(triton.cdiv(OUT_STORAGE_SIZE, zero_block),)](
        out,
        OUT_STORAGE_SIZE,
        BLOCK=zero_block,
    )
    _longformer_softmax_layout_kernel[(ROWS,)](
        query_mask,
        bmm,
        key_mask,
        out,
        BLOCK_N=1024,
        R=RNUMEL,
        SEQ=SEQ_LEN,
        HEADS=N_HEADS,
        CHUNK=LOCAL_CHUNK,
        PADDED_R=PADDED_RNUMEL,
        FINAL_INNER_=FINAL_INNER,
        OUT_D_=OUT_D,
        OUT_M_STRIDE=OUT_STRIDE[0],
        OUT_T_STRIDE=OUT_STRIDE[1],
        num_warps=4,
    )
    return out


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Compute the full Repro.forward output for this Longformer inference row."""
    query_mask, bmm, key_mask = inputs[:3]
    return _longformer_softmax_layout(query_mask, bmm, key_mask)


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
def run_bench(warmup: int, rep: int) -> list[float]:
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
