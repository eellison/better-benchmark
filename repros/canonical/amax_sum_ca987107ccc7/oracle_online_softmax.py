"""
Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full GPT-Neo attention-mask softmax materialization with one Triton program per `[128]` row, recomputing the causal/segment predicate from `unsqueeze` and `cumsum`, loading the sliced `arg8_1[:, :, :128, :128]` attention mask directly, applying the same two `where` masks to `bmm.view(32, 16, 128, 128)`, and normalizing the row without materializing either mask tensor or the masked-score intermediate. Inductor lowers the decomposed graph as generic mask construction, two large `where` operations, an add, and a softmax reduction over the materialized `[32, 16, 128, 128]` score tensor; it cannot currently see this as one masked softmax row template because the predicate is built from advanced indexing into `cumsum` plus a separate broadcasted attention mask slice. The fix is a NEW_PATTERN lowering for attention masked softmax that accepts structured mask predicates and emits a row kernel that recomputes cheap boolean masks at the point of use.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_ca987107ccc7"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 16
SEQ = 128
ATTN_MASK_STRIDE = 2048
N_ROWS = BATCH * HEADS * SEQ

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _masked_softmax_rows_kernel(
    unsqueeze_ptr,
    cumsum_ptr,
    bmm_ptr,
    arg8_ptr,
    out_ptr,
    block_n: tl.constexpr,
    seq_len: tl.constexpr,
    n_heads: tl.constexpr,
    attn_mask_stride: tl.constexpr,
):
    row = tl.program_id(0)
    q = row % seq_len
    batch_head = row // seq_len
    batch = batch_head // n_heads

    cols = tl.arange(0, block_n)
    col_mask = cols < seq_len

    q_index = tl.load(unsqueeze_ptr + q)
    k_index = tl.load(unsqueeze_ptr + cols, mask=col_mask, other=0)

    causal_mask = k_index <= q_index
    q_segment = tl.load(cumsum_ptr + batch * seq_len + q_index)
    k_segment = tl.load(
        cumsum_ptr + batch * seq_len + k_index,
        mask=col_mask,
        other=q_segment + 1,
    )
    segment_mask = causal_mask & (k_segment == q_segment)

    score_offsets = row * seq_len + cols
    scores = tl.load(bmm_ptr + score_offsets, mask=col_mask, other=0.0).to(tl.float32)

    attn_mask = tl.load(
        arg8_ptr + q * attn_mask_stride + cols,
        mask=col_mask,
        other=0,
    ).to(tl.int1)

    min_f32 = tl.full((block_n,), -3.4028234663852886e38, tl.float32)
    zeros = tl.zeros((block_n,), tl.float32)
    masked_scores = tl.where(attn_mask, scores, min_f32) + tl.where(
        segment_mask,
        zeros,
        min_f32,
    )

    row_max = tl.max(masked_scores, axis=0)
    numer = tl.exp(masked_scores - row_max)
    denom = tl.sum(numer, axis=0)
    out = numer / denom

    tl.store(out_ptr + score_offsets, out, mask=col_mask)


def _launch_oracle(
    unsqueeze: torch.Tensor,
    cumsum: torch.Tensor,
    bmm: torch.Tensor,
    arg8_1: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert unsqueeze.is_cuda and cumsum.is_cuda and bmm.is_cuda and arg8_1.is_cuda
    assert unsqueeze.dtype == torch.int64 and unsqueeze.shape == (1, SEQ)
    assert cumsum.dtype == torch.int64 and cumsum.shape == (BATCH, SEQ)
    assert bmm.dtype == torch.float32 and bmm.shape == (BATCH * HEADS, SEQ, SEQ)
    assert arg8_1.dtype == torch.bool and arg8_1.shape == (1, 1, 2048, 2048)
    assert out.dtype == torch.float32 and out.shape == bmm.shape

    _masked_softmax_rows_kernel[(N_ROWS,)](
        unsqueeze,
        cumsum,
        bmm,
        arg8_1,
        out,
        block_n=block_n,
        seq_len=SEQ,
        n_heads=HEADS,
        attn_mask_stride=ATTN_MASK_STRIDE,
        num_warps=num_warps,
    )
    return out


def oracle_attention_masked_softmax(
    unsqueeze: torch.Tensor,
    cumsum: torch.Tensor,
    bmm: torch.Tensor,
    arg8_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    _shape_param_3=None,
    *,
    block_n: int = 128,
    num_warps: int = 4,
) -> torch.Tensor:
    out = torch.empty_strided(
        bmm.shape,
        bmm.stride(),
        device=bmm.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        unsqueeze,
        cumsum,
        bmm,
        arg8_1,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_inputs(module, seed: int) -> tuple:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)


def _bench_cuda(fn, warmup: int, rep: int) -> float:
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
    return times[len(times) // 2]


def _compile_with_config(module, inputs: tuple, config: dict[str, object], warmup: int):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_check(block_n: int, num_warps: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
    model = module.Repro().cuda()

    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_attention_masked_softmax(
            *inputs,
            block_n=block_n,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

    diff = (got - ref).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    max_rel = torch.nan_to_num(diff / ref.abs().clamp_min(1e-8), nan=0.0).max().item()
    stride_match = got.stride() == ref.stride()
    ok = torch.allclose(got, ref, rtol=1e-5, atol=1e-6, equal_nan=True) and stride_match

    print(
        "check GPT-Neo masked softmax materialization: "
        f"output={tuple(got.shape)} max_abs={max_abs:.6e} "
        f"max_rel={max_rel:.6e} stride_match={stride_match} allclose={ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(block_n: int, num_warps: int, warmup: int, rep: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    out = torch.empty_strided(
        inputs[2].shape,
        inputs[2].stride(),
        device=inputs[2].device,
        dtype=torch.float32,
    )

    traffic_bytes = (
        N_ROWS * SEQ * 4
        + N_ROWS * SEQ * 4
        + BATCH * SEQ * 8
        + SEQ * 8
        + SEQ * SEQ
    )

    print(
        "oracle shape: "
        f"unsqueeze=i64[1,{SEQ}], cumsum=i64[{BATCH},{SEQ}], "
        f"bmm=f32[{BATCH * HEADS},{SEQ},{SEQ}], arg8_1=bool[1,1,2048,2048]"
    )
    print(f"single-kernel logical traffic: {traffic_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                out,
                block_n=block_n,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = traffic_bytes / (oracle_us * 1e-6) / 1e12
    print(f"oracle full-scope masked softmax: {oracle_us:.3f} us ({oracle_bw:.3f} TB/s)")

    if no_compile:
        return

    print("torch.compile full repro timings include mask construction, where/add, softmax, and output view")
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module, inputs, config, warmup)
            us = _bench_cuda(lambda: compiled(*inputs), warmup=warmup, rep=rep)
            print(f"torch.compile {label}: {us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-n", type=int, default=128, help="Triton row tile size")
    parser.add_argument("--num-warps", type=int, default=4, help="Triton warps per row")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check(block_n=args.block_n, num_warps=args.num_warps)
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            block_n=args.block_n,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
