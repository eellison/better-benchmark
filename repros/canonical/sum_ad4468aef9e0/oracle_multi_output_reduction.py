"""
Full-scope Triton oracle for sum_ad4468aef9e0 (GPT-2 QKV concat sum).

The compiled repro permutes three strided `[32, 12, 512, 64]` tensors into
contiguous `[32, 512, 768]` views, concatenates them as `[q, v, k]`, flattens to
`[16384, 2304]`, and returns the column sum. This oracle consumes the same three
input tensors plus shape parameters and returns the same contiguous
`float32[2304]` output. It covers the full computation scope: the timed Triton
path directly reduces the three original strided tensors and does not benchmark
only a reduction subset over a pre-materialized cat.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle differs from
Inductor by indexing the original Q/K/V-style strided inputs and writing the
three concatenated reduction segments directly, skipping the logical
permute/view/cat materialization. Inductor can already lower this graph as a
fused reduction over the original inputs, so there is no separate missing
multi-output fusion, scatter-reduce, algebraic-elimination, or recompute-fusion
opportunity exposed by this repro; if the oracle does not beat the two required
local compile configs and the historical best, the relevant conclusion is
diagnosis-only/BANDWIDTH_BOUND. The Inductor change that would help is narrower
tuning of the existing cat-sum reduction template for this memory layout rather
than a new scheduler primitive.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_ad4468aef9e0"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 12
SEQ = 512
HEAD_DIM = 64
QKV_CHUNK = HEADS * HEAD_DIM
OUT_COLS = 3 * QKV_CHUNK
ROWS = BATCH * SEQ

INPUT_STRIDE_B = SEQ * HEADS * HEAD_DIM
INPUT_STRIDE_H = HEAD_DIM
INPUT_STRIDE_S = HEADS * HEAD_DIM

HISTORICAL_BEST_COMPILE_US = 37.82400116324425
DEFAULT_MODE = "split"
DEFAULT_BLOCK_R = 512
DEFAULT_BLOCK_C = 32
DEFAULT_PARTIAL_CHUNKS = 32
DEFAULT_NUM_WARPS = 4

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _direct_qkv_sum_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    ROWS_: tl.constexpr,
    QKV_CHUNK_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    col_base = tl.program_id(0) * BLOCK_C
    rows = tl.arange(0, BLOCK_R)
    cols = col_base + tl.arange(0, BLOCK_C)
    col_mask = cols < QKV_CHUNK_

    acc_q = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for row_base in tl.range(0, ROWS_, BLOCK_R):
        r = row_base + rows
        row_mask = r < ROWS_
        offsets = r[:, None] * QKV_CHUNK_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        acc_q += tl.sum(tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        acc_k += tl.sum(tl.load(k_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        acc_v += tl.sum(tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

    tl.store(out_ptr + cols, acc_q, mask=col_mask)
    tl.store(out_ptr + QKV_CHUNK_ + cols, acc_v, mask=col_mask)
    tl.store(out_ptr + 2 * QKV_CHUNK_ + cols, acc_k, mask=col_mask)


@triton.jit
def _split_qkv_sum_partials_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    QKV_CHUNK_: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    col_block = tl.program_id(0)
    chunk = tl.program_id(1)
    col_base = col_block * BLOCK_C
    rows = tl.arange(0, BLOCK_R)
    cols = col_base + tl.arange(0, BLOCK_C)
    col_mask = cols < QKV_CHUNK_

    chunk_start = chunk * CHUNK_SIZE
    acc_q = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for chunk_r_base in tl.range(0, CHUNK_SIZE, BLOCK_R):
        chunk_r = chunk_r_base + rows
        r = chunk_start + chunk_r
        row_mask = (chunk_r < CHUNK_SIZE) & (r < ROWS_)
        offsets = r[:, None] * QKV_CHUNK_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        acc_q += tl.sum(tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        acc_k += tl.sum(tl.load(k_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        acc_v += tl.sum(tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

    partial_base = chunk * QKV_CHUNK_ + cols
    tl.store(partials_ptr + partial_base, acc_q, mask=col_mask)
    tl.store(
        partials_ptr + NUM_PARTIALS * QKV_CHUNK_ + partial_base,
        acc_v,
        mask=col_mask,
    )
    tl.store(
        partials_ptr + 2 * NUM_PARTIALS * QKV_CHUNK_ + partial_base,
        acc_k,
        mask=col_mask,
    )


@triton.jit
def _finalize_qkv_sum_kernel(
    partials_ptr,
    out_ptr,
    QKV_CHUNK_: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    col_base = tl.program_id(0) * BLOCK_C
    cols = col_base + tl.arange(0, BLOCK_C)
    partials = tl.arange(0, BLOCK_PARTIALS)
    col_mask = cols < QKV_CHUNK_
    partial_mask = partials < NUM_PARTIALS
    mask = partial_mask[:, None] & col_mask[None, :]
    offsets = partials[:, None] * QKV_CHUNK_ + cols[None, :]

    q_vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    v_vals = tl.load(
        partials_ptr + NUM_PARTIALS * QKV_CHUNK_ + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    k_vals = tl.load(
        partials_ptr + 2 * NUM_PARTIALS * QKV_CHUNK_ + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    tl.store(out_ptr + cols, tl.sum(q_vals, axis=0), mask=col_mask)
    tl.store(out_ptr + QKV_CHUNK_ + cols, tl.sum(v_vals, axis=0), mask=col_mask)
    tl.store(out_ptr + 2 * QKV_CHUNK_ + cols, tl.sum(k_vals, axis=0), mask=col_mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def _validate_tiling(mode: str, block_r: int, block_c: int, partial_chunks: int, num_warps: int) -> None:
    if mode not in {"direct", "split"}:
        raise ValueError("--mode must be 'direct' or 'split'")
    for name, value in (("block_r", block_r), ("block_c", block_c)):
        if value <= 0 or (value & (value - 1)) != 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive power of two")
    if partial_chunks <= 0:
        raise ValueError("--partial-chunks must be positive")
    if num_warps <= 0:
        raise ValueError("--num-warps must be positive")


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor | None = None) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != (BATCH, HEADS, SEQ, HEAD_DIM):
            raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} has unexpected dtype {tensor.dtype}")
        if tuple(tensor.stride()) != (INPUT_STRIDE_B, INPUT_STRIDE_H, INPUT_STRIDE_S, 1):
            raise ValueError(f"{name} has unexpected stride {tensor.stride()}")
    if out is not None:
        if tuple(out.shape) != (OUT_COLS,) or out.stride() != (1,) or out.dtype != torch.float32:
            raise ValueError("unexpected output buffer layout")


def _chunk_size(partial_chunks: int) -> int:
    return triton.cdiv(ROWS, partial_chunks)


def _make_workspace(
    device: torch.device,
    mode: str,
    partial_chunks: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    partials = None
    if mode == "split":
        partials = torch.empty((3, partial_chunks, QKV_CHUNK), device=device, dtype=torch.float32)
    out = torch.empty_strided((OUT_COLS,), (1,), device=device, dtype=torch.float32)
    return partials, out


def _oracle_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    partials: torch.Tensor | None,
    out: torch.Tensor,
    *,
    mode: str,
    block_r: int,
    block_c: int,
    partial_chunks: int,
    num_warps: int,
) -> torch.Tensor:
    _validate_tiling(mode, block_r, block_c, partial_chunks, num_warps)
    _validate_inputs(q, k, v, out)

    grid_cols = triton.cdiv(QKV_CHUNK, block_c)
    if mode == "direct":
        _direct_qkv_sum_kernel[(grid_cols,)](
            q,
            k,
            v,
            out,
            ROWS_=ROWS,
            QKV_CHUNK_=QKV_CHUNK,
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
        return out

    if partials is None:
        raise ValueError("split mode requires partials workspace")
    if tuple(partials.shape) != (3, partial_chunks, QKV_CHUNK) or partials.dtype != torch.float32:
        raise ValueError("unexpected partials workspace layout")

    _split_qkv_sum_partials_kernel[(grid_cols, partial_chunks)](
        q,
        k,
        v,
        partials,
        ROWS_=ROWS,
        QKV_CHUNK_=QKV_CHUNK,
        CHUNK_SIZE=_chunk_size(partial_chunks),
        NUM_PARTIALS=partial_chunks,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    _finalize_qkv_sum_kernel[(grid_cols,)](
        partials,
        out,
        QKV_CHUNK_=QKV_CHUNK,
        NUM_PARTIALS=partial_chunks,
        BLOCK_PARTIALS=triton.next_power_of_2(partial_chunks),
        BLOCK_C=block_c,
        num_warps=1,
    )
    return out


def oracle_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
    _shape_param_4: object,
    *,
    mode: str = DEFAULT_MODE,
    block_r: int = DEFAULT_BLOCK_R,
    block_c: int = DEFAULT_BLOCK_C,
    partial_chunks: int = DEFAULT_PARTIAL_CHUNKS,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4
    partials, out = _make_workspace(q.device, mode, partial_chunks)
    return _oracle_into(
        q,
        k,
        v,
        partials,
        out,
        mode=mode,
        block_r=block_r,
        block_c=block_c,
        partial_chunks=partial_chunks,
        num_warps=num_warps,
    )


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    return oracle_fused(*inputs)


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return float(diff.max().item()), float(rel.max().item())


def run_check(
    rtol: float,
    atol: float,
    mode: str,
    block_r: int,
    block_c: int,
    partial_chunks: int,
    num_warps: int,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_output(inputs)
        actual = oracle_fused(
            *inputs,
            mode=mode,
            block_r=block_r,
            block_c=block_c,
            partial_chunks=partial_chunks,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

    shape_ok = actual.shape == ref.shape
    dtype_ok = actual.dtype == ref.dtype
    stride_ok = actual.stride() == ref.stride()
    max_abs, max_rel = _max_diff(actual, ref)
    close_ok = torch.allclose(actual.float(), ref.float(), rtol=rtol, atol=atol)
    ok = shape_ok and dtype_ok and stride_ok and close_ok
    print(
        f"output[0]: shape={list(actual.shape)} dtype={actual.dtype} stride={actual.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={close_ok} "
        f"shape_match={shape_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    rep: int,
    warmup: int,
    no_compile: bool,
    mode: str,
    block_r: int,
    block_c: int,
    partial_chunks: int,
    num_warps: int,
) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    q, k, v = inputs[:3]
    partials, out = _make_workspace(q.device, mode, partial_chunks)

    timings: dict[str, float] = {}
    with torch.no_grad():
        _oracle_into(
            q,
            k,
            v,
            partials,
            out,
            mode=mode,
            block_r=block_r,
            block_c=block_c,
            partial_chunks=partial_chunks,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: _oracle_into(
                q,
                k,
                v,
                partials,
                out,
                mode=mode,
                block_r=block_r,
                block_c=block_c,
                partial_chunks=partial_chunks,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_fused"] = oracle_us
    print(
        f"oracle_fused full-scope qkv cat-sum: {oracle_us:.3f} us "
        f"(mode={mode}, block_r={block_r}, block_c={block_c}, "
        f"partial_chunks={partial_chunks}, num_warps={num_warps})"
    )

    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            model = module.Repro().cuda()
            with torch.no_grad():
                compiled = _compile_with_config(model, inputs, config)
                compiled_us = triton.testing.do_bench(
                    lambda: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                ) * 1000.0
            timings[label] = compiled_us
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    best_required_compile = min(
        (timings[label] for label, _ in COMPILE_CONFIGS if label in timings),
        default=float("inf"),
    )
    best_gate = min(best_required_compile, HISTORICAL_BEST_COMPILE_US)
    valid_floor = oracle_us < best_gate
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if valid_floor else 'no (diagnosis-only)'}")
    print(
        json.dumps(
            {
                "repro_id": REPRO_ID,
                "oracle_us": round(oracle_us, 3),
                "compile_us": (
                    round(best_required_compile, 3)
                    if best_required_compile != float("inf")
                    else None
                ),
                "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
                "ratio": (
                    round(best_required_compile / oracle_us, 6)
                    if best_required_compile != float("inf")
                    else None
                ),
                "status": "GOOD" if valid_floor else "DIAGNOSIS_ONLY",
                "classification": "BANDWIDTH_BOUND",
            }
        )
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=3e-3)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--mode", choices=("direct", "split"), default=DEFAULT_MODE)
    parser.add_argument("--block-r", type=int, default=DEFAULT_BLOCK_R)
    parser.add_argument("--block-c", type=int, default=DEFAULT_BLOCK_C)
    parser.add_argument("--partial-chunks", type=int, default=DEFAULT_PARTIAL_CHUNKS)
    parser.add_argument("--num-warps", type=int, default=DEFAULT_NUM_WARPS)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if args.rep <= 0:
        raise ValueError("--rep must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        mode=args.mode,
        block_r=args.block_r,
        block_c=args.block_c,
        partial_chunks=args.partial_chunks,
        num_warps=args.num_warps,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            rep=args.rep,
            warmup=args.warmup,
            no_compile=args.no_compile,
            mode=args.mode,
            block_r=args.block_r,
            block_c=args.block_c,
            partial_chunks=args.partial_chunks,
            num_warps=args.num_warps,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
