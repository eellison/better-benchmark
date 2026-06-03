"""
Oracle for argmax_mean_180a59791f51: Reformer LSH routing and post-sort gather.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the Reformer LSH
route key directly as a signed argmax over the original 64-vector, using max(x)
and max(-x) to produce the same 0..127 bucket without first materializing the
128-wide `cat([x, -x])`, then sorts `bucket * 4096 + position` and applies the
same two sorted gathers, RMS scaling, and cyclic concatenations as the captured
graph; Inductor currently emits a pointwise kernel that writes the full
cat/negation buffer, a separate argmax/key kernel, an external sort, and
post-sort kernels because the scheduler does not canonicalize
`argmax(cat([x, -x]))` plus bounded-key sort arithmetic into an LSH routing
operation and the external sort boundary prevents this algebra from being
pulled into the consumers. The Inductor fix is a NEW_PATTERN lowering for this
Reformer LSH routing idiom: signed-abs argmax key generation, bounded-key
position sort, and fused sorted gather/RMS/cyclic-concat epilogue without the
materialized cat input.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton


REPRO_ID = "argmax_mean_180a59791f51"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
SEQ = 4096
DIM = 64
CHUNKS = 64
SIGNED_BUCKETS = 128
FLAT_CHUNKS = BATCH * HEADS * CHUNKS

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


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _signed_lsh_bucket(bmm: torch.Tensor) -> torch.Tensor:
    x = (
        bmm.view(HEADS, BATCH, SEQ, 1, 1, DIM)
        .permute(1, 0, 3, 2, 5, 4)
        .reshape(BATCH, HEADS, SEQ, DIM)
    )
    pos_val, pos_idx = x.max(dim=-1)
    neg_val, neg_idx = (-x).max(dim=-1)
    return torch.where(pos_val >= neg_val, pos_idx, neg_idx + DIM).to(torch.int64)


def oracle_reformer_lsh_routing(
    bmm: torch.Tensor,
    permute_1: torch.Tensor,
    mm_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bucket = _signed_lsh_bucket(bmm)
    positions = torch.arange(SEQ, device=bmm.device, dtype=torch.int64).view(1, 1, SEQ)
    keys = bucket * SEQ + positions
    sorted_keys, sorted_pos = torch.sort(keys, dim=-1)
    gather_index = sorted_pos.unsqueeze(-1).expand(BATCH, HEADS, SEQ, DIM)

    gathered = torch.gather(permute_1, 2, gather_index)
    gathered_view = gathered.view(BATCH, HEADS, CHUNKS, CHUNKS, DIM)
    gathered_out = gathered_view.view(FLAT_CHUNKS, CHUNKS, DIM)

    rms_scale = torch.rsqrt((gathered_view * gathered_view).mean(dim=-1, keepdim=True) + 1e-6) / 8.0
    normalized = gathered_view * rms_scale
    shifted = torch.cat([normalized[:, :, -1:, :, :], normalized[:, :, :-1, :, :]], dim=2)
    normalized_out = (
        torch.cat([shifted, normalized], dim=3)
        .permute(0, 1, 2, 4, 3)
        .reshape(FLAT_CHUNKS, DIM, SIGNED_BUCKETS)
    )

    mm_view = mm_1.view(BATCH, SEQ, HEADS, DIM).permute(0, 2, 1, 3)
    gathered_mm = torch.gather(mm_view, 2, gather_index)
    gathered_mm_view = gathered_mm.view(BATCH, HEADS, CHUNKS, CHUNKS, DIM)
    shifted_mm = torch.cat([gathered_mm_view[:, :, -1:, :, :], gathered_mm_view[:, :, :-1, :, :]], dim=2)
    mm_out = torch.cat([shifted_mm, gathered_mm_view], dim=3).reshape(FLAT_CHUNKS, SIGNED_BUCKETS, DIM)

    return (
        sorted_keys,
        bucket.view(BATCH, HEADS, 1, SEQ),
        gathered_out,
        normalized_out,
        mm_out,
    )


class ReformerLSHRoutingOracle(torch.nn.Module):
    def forward(
        self,
        bmm: torch.Tensor,
        permute_1: torch.Tensor,
        mm_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return oracle_reformer_lsh_routing(bmm, permute_1, mm_1)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bmm, permute_1, mm_1, *_shape_params = inputs
    return bmm, permute_1, mm_1


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    if actual.dtype.is_floating_point:
        diff = (actual.float() - expected.float()).abs()
        rel = diff / (expected.float().abs() + 1e-8)
        return diff.max().item(), rel.max().item()

    diff = (actual.to(torch.int64) - expected.to(torch.int64)).abs()
    return diff.max().item(), 0.0


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)

    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_reformer_lsh_routing(*oracle_inputs)
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        if got.dtype.is_floating_point:
            output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        else:
            output_ok = torch.equal(got, expected)
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={list(got.stride())} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={output_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            fn()
        torch.cuda.synchronize()
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        with torch.no_grad():
            for _ in range(3):
                compiled(*inputs)
            torch.cuda.synchronize()
    return compiled


def run_bench(warmup: int, rep: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the oracle benchmark")

    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    oracle_model = ReformerLSHRoutingOracle().cuda()

    logical_read_bytes = BATCH * HEADS * SEQ * DIM * 4 * 3
    logical_write_bytes = (
        BATCH * HEADS * SEQ * 8
        + BATCH * HEADS * SEQ * 8
        + FLAT_CHUNKS * CHUNKS * DIM * 4
        + FLAT_CHUNKS * DIM * SIGNED_BUCKETS * 4
        + FLAT_CHUNKS * SIGNED_BUCKETS * DIM * 4
    )
    print(f"oracle shape: bmm=f32[{HEADS}, {BATCH * SEQ}, {DIM}], seq={SEQ}, dim={DIM}")
    print(f"direct signed-routing logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    eager_us = _bench_cuda(lambda: oracle_reformer_lsh_routing(*oracle_inputs), warmup=warmup, rep=rep)
    print(f"oracle eager direct-routing: {eager_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    for label, config in COMPILE_CONFIGS:
        try:
            compiled_oracle = _compile_with_config(oracle_model, oracle_inputs, config)
            oracle_us = _bench_cuda(lambda: compiled_oracle(*oracle_inputs), warmup=warmup, rep=rep)
            print(f"oracle torch.compile {label}: {oracle_us:.3f} us")
        except Exception as exc:
            print(f"oracle torch.compile {label}: FAILED ({exc})")

        try:
            compiled_repro = _compile_with_config(module.Repro().cuda(), inputs, config)
            repro_us = _bench_cuda(lambda: compiled_repro(*inputs), warmup=warmup, rep=rep)
            print(f"torch.compile repro {label}: {repro_us:.3f} us")
        except Exception as exc:
            print(f"torch.compile repro {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile oracle and repro timings")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
