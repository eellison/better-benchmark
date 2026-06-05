"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer sliding-window score assembly from the captured bmm/base/mask/bias inputs, fuses softmax, query-mask fill, Inductor-RNG dropout, and final strided output layout in a custom Triton path, whereas Inductor currently lowers the graph as separate view/pad/slice/scatter assembly, generic amax/exp/sum/div softmax, random dropout, and layout kernels; Inductor cannot do this today because the scheduler has no Longformer attention pattern that recognizes the skewed chunk assembly plus first/last chunk masks and fuses it with the reduction epilogue and destination-layout scatter; the fix is NEW_PATTERN: add a Longformer sliding-window attention lowering that canonicalizes the structured band assembly and emits one fused softmax/dropout/layout kernel."""
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "amax_sum_87e1fb077f24"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ_LEN = 1024
N_HEADS = 12
BH = BATCH * N_HEADS
LOCAL_CHUNK = 256
CHUNKS = 4
RNUMEL = 513
PADDED_RNUMEL = 770
FINAL_INNER = 769
OUT_M = BH * CHUNKS
OUT_D = 768
OUT_T = 256
OUT_SHAPE = (OUT_M, OUT_D, OUT_T)
OUT_STRIDE = (LOCAL_CHUNK * PADDED_RNUMEL, 1, FINAL_INNER)
OUT_STORAGE_SIZE = (OUT_M - 1) * OUT_STRIDE[0] + (OUT_D - 1) + (OUT_T - 1) * OUT_STRIDE[2] + 1
ROWS = BATCH * SEQ_LEN * N_HEADS
BLOCK_N = 1024
SEED_INDEX = 33

SLICE3_BH_STRIDE = 525312
SLICE3_CHUNK_STRIDE = 131328
SLICE3_POS_STRIDE = 513
PERMUTE12_BATCH_STRIDE = 789504
PERMUTE12_POS_STRIDE = 257
PERMUTE12_HEAD_STRIDE = 65792


@lru_cache(maxsize=1)
def _load_repro_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs() -> tuple:
    return _load_repro_module().make_inputs()


def get_repro_instance() -> torch.nn.Module:
    return _load_repro_module().Repro().eval()


def _clone_inputs(inputs: tuple) -> tuple:
    cloned = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            if item.dim() == 0:
                cloned.append(item.clone())
            else:
                copy = torch.empty_strided(
                    tuple(item.shape),
                    tuple(item.stride()),
                    device=item.device,
                    dtype=item.dtype,
                )
                copy.copy_(item)
                cloned.append(copy)
        else:
            cloned.append(item)
    return tuple(cloned)


def _inductor_random_like_repro(inductor_seeds: torch.Tensor) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
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
def _longformer_full_kernel(
    bmm_ptr,
    slice3_ptr,
    first_mask_ptr,
    first_last_value_ptr,
    last_mask_ptr,
    global_bias_ptr,
    query_mask_ptr,
    full2_ptr,
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
    SLICE3_BH_STRIDE_: tl.constexpr,
    SLICE3_POS_STRIDE_: tl.constexpr,
    PERMUTE12_BATCH_STRIDE_: tl.constexpr,
    PERMUTE12_POS_STRIDE_: tl.constexpr,
    PERMUTE12_HEAD_STRIDE_: tl.constexpr,
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
    from_right = col_i32 >= CHUNK
    source_chunk_right = tl.minimum(chunk_id, 2)
    skew_row_right = tl.where(chunk_id == 3, pos + 256, pos)
    source_chunk_left = tl.where(chunk_id == 0, 0, chunk_id - 1)
    skew_row_left = tl.where(chunk_id == 0, pos - 1, pos + 255)
    source_chunk = tl.where(from_right, source_chunk_right, source_chunk_left)
    skew_row = tl.where(from_right, skew_row_right, skew_row_left)
    skew_col = tl.where(from_right, col_i32 - 256, col_i32 + 257)
    first_left_interior = (chunk_id == 0) & (pos > 0) & (col_i32 > 0) & (col_i32 < 256)
    has_bmm_source = from_right | (chunk_id != 0) | first_left_interior

    bmm_score = _load_skewed_bmm(
        bmm_ptr,
        bh,
        source_chunk,
        skew_row,
        skew_col,
        valid_cols & has_bmm_source,
    )

    safe_col = tl.minimum(col_i32, R - 1)
    first_chunk_base = tl.load(
        slice3_ptr + bh * SLICE3_BH_STRIDE_ + pos * SLICE3_POS_STRIDE_ + safe_col,
        mask=valid_cols,
        other=0.0,
    ).to(tl.float32)
    first_select = (chunk_id == 0) & ((pos == 0) | (col_i32 == 0)) & (col_i32 < 256)
    local_score = tl.where(first_select, first_chunk_base, bmm_score)

    first_mask_offsets = ((batch * CHUNK + pos) * HEADS + head) * 257 + tl.minimum(col_i32, 256)
    first_value_offsets = (
        batch * PERMUTE12_BATCH_STRIDE_
        + pos * PERMUTE12_POS_STRIDE_
        + head * PERMUTE12_HEAD_STRIDE_
        + tl.minimum(col_i32, 256)
    )
    first_mask = tl.load(
        first_mask_ptr + first_mask_offsets,
        mask=(chunk_id == 0) & (col_i32 < 257),
        other=0,
    ) != 0
    first_value = tl.load(
        first_last_value_ptr + first_value_offsets,
        mask=(chunk_id == 0) & (col_i32 < 257),
        other=0.0,
    ).to(tl.float32)
    local_score = tl.where((chunk_id == 0) & (col_i32 < 257) & first_mask, first_value, local_score)

    last_col = col_i32 - 256
    safe_last_col = tl.maximum(tl.minimum(last_col, 256), 0)
    last_mask_offsets = ((batch * CHUNK + pos) * HEADS + head) * 257 + safe_last_col
    last_value_offsets = (
        batch * PERMUTE12_BATCH_STRIDE_
        + pos * PERMUTE12_POS_STRIDE_
        + head * PERMUTE12_HEAD_STRIDE_
        + safe_last_col
    )
    last_mask = tl.load(
        last_mask_ptr + last_mask_offsets,
        mask=(chunk_id == 3) & (col_i32 >= 256) & valid_cols,
        other=0,
    ) != 0
    last_value = tl.load(
        first_last_value_ptr + last_value_offsets,
        mask=(chunk_id == 3) & (col_i32 >= 256) & valid_cols,
        other=0.0,
    ).to(tl.float32)
    local_score = tl.where((chunk_id == 3) & (col_i32 >= 256) & last_mask, last_value, local_score)

    bias = tl.load(
        global_bias_ptr + (batch * SEQ + seq) * R + safe_col,
        mask=valid_cols,
        other=0.0,
    ).to(tl.float32)
    scores = tl.where(valid_cols, local_score + bias, -float("inf"))

    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    values = numer / denom

    query_masked = tl.load(query_mask_ptr + batch * SEQ + seq) != 0
    full2 = tl.load(full2_ptr).to(tl.float32)
    values = tl.where(query_masked, full2, values)

    keep_dropout = tl.load(random_ptr + row * R + safe_col, mask=valid_cols, other=0.0) > 1.0e-30
    values = tl.where(keep_dropout, values, 0.0)

    padded_linear = pos * PADDED_R + safe_col
    out_t = padded_linear // FINAL_R
    out_d = padded_linear - out_t * FINAL_R
    out_m = bh * CHUNKS_ + chunk_id
    out_offsets = out_m * OUT_M_STRIDE + out_d + out_t * OUT_T_STRIDE
    store_mask = valid_cols & (out_d < OUT_D_)
    tl.store(out_ptr + out_offsets, values, mask=store_mask)


def _launch_oracle(
    bmm_22: torch.Tensor,
    slice_3: torch.Tensor,
    convert_element_type: torch.Tensor,
    permute_12: torch.Tensor,
    convert_element_type_1: torch.Tensor,
    permute_25: torch.Tensor,
    unsqueeze_11: torch.Tensor,
    full_2: torch.Tensor,
    inductor_seeds: torch.Tensor,
) -> torch.Tensor:
    random_values = _inductor_random_like_repro(inductor_seeds)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_22.device,
        dtype=bmm_22.dtype,
    )
    _zero_kernel[(triton.cdiv(OUT_STORAGE_SIZE, BLOCK_N),)](
        out,
        OUT_STORAGE_SIZE,
        BLOCK=BLOCK_N,
    )
    _longformer_full_kernel[(ROWS,)](
        bmm_22,
        slice_3,
        convert_element_type,
        permute_12,
        convert_element_type_1,
        permute_25,
        unsqueeze_11,
        full_2,
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
        SLICE3_BH_STRIDE_=SLICE3_BH_STRIDE,
        SLICE3_POS_STRIDE_=SLICE3_POS_STRIDE,
        PERMUTE12_BATCH_STRIDE_=PERMUTE12_BATCH_STRIDE,
        PERMUTE12_POS_STRIDE_=PERMUTE12_POS_STRIDE,
        PERMUTE12_HEAD_STRIDE_=PERMUTE12_HEAD_STRIDE,
        OUT_M_STRIDE=OUT_STRIDE[0],
        OUT_T_STRIDE=OUT_STRIDE[2],
        num_warps=4,
    )
    return out


def oracle_forward(inputs: tuple) -> torch.Tensor:
    bmm_22 = inputs[0]
    slice_3 = inputs[2]
    convert_element_type = inputs[4]
    permute_12 = inputs[5]
    convert_element_type_1 = inputs[6]
    permute_25 = inputs[7]
    unsqueeze_11 = inputs[8]
    full_2 = inputs[9]
    inductor_seeds = inputs[10]
    return _launch_oracle(
        bmm_22,
        slice_3,
        convert_element_type,
        permute_12,
        convert_element_type_1,
        permute_25,
        unsqueeze_11,
        full_2,
        inductor_seeds,
    )


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
