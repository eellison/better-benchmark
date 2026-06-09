"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5 additive-bias attention softmax/dropout return from Repro.forward by fusing the [64,1024,1024] to [8,8,1024,1024] view, full bias add, stable last-dimension softmax, exact Inductor seed-index-59 RNG dropout, scale, expand/view, and final transposed [64,1024,1024] output view into one persistent row Triton softmax kernel, whereas Inductor currently lowers the decomposed view/add/amax/sub/exp/sum/div/inductor_random/gt/mul/expand/view/permute graph as generic reduction, pointwise RNG/dropout, and layout kernels over materialized softmax intermediates; Inductor cannot do this today because its scheduler/pattern matcher does not canonicalize additive-bias attention softmax with stochastic Inductor RNG dropout and a trailing layout-only transpose into a single persistent online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering that recognizes last-dimension additive attention softmax with RNG dropout and fuses the bias, normalization, dropout scale, and output-layout epilogue into the row-softmax schedule."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

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



REPRO_ID = "amax_sum_ab3238e72dc2"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_hf_t5_train_000_04883ac2"

BATCH = 8
N_HEADS = 8
Q_LEN = 1024
K_LEN = 1024
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 59



@triton.jit
def _bias_softmax_dropout_kernel(
    bmm_ptr,
    bias_ptr,
    random_ptr,
    out_base_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    offsets = row * K + cols

    bmm = tl.load(bmm_ptr + offsets, mask=col_mask, other=-float("inf")).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    scores = bmm + bias

    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    softmax = numer / denom

    random = tl.load(random_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    keep = random > DROPOUT_P_CONST
    out = tl.where(keep, softmax * DROPOUT_SCALE_CONST, 0.0)

    tl.store(out_base_ptr + offsets, out, mask=col_mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda()
        if isinstance(value, torch.Tensor) and value.device.type != "cuda"
        else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _check_shape_params(
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
) -> None:
    assert list(_shape_param_0) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_1) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_2) == [BH, Q_LEN, K_LEN]


def _inductor_random_like_repro(inductor_seeds: torch.Tensor) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    return torch.ops.prims.inductor_random.default(
        [BATCH, N_HEADS, Q_LEN, K_LEN],
        seed,
        "rand",
    )


def _launch_oracle(
    bmm_34: torch.Tensor,
    add_44: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    assert bmm_34.is_cuda and add_44.is_cuda and random_values.is_cuda
    assert out_base.is_cuda
    assert bmm_34.dtype == torch.float32 and bmm_34.shape == (BH, Q_LEN, K_LEN)
    assert add_44.dtype == torch.float32 and add_44.shape == (
        BATCH,
        N_HEADS,
        Q_LEN,
        K_LEN,
    )
    assert random_values.dtype == torch.float32
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert out_base.dtype == torch.float32 and out_base.shape == bmm_34.shape
    assert bmm_34.is_contiguous()
    assert add_44.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    _bias_softmax_dropout_kernel[(N_ROWS,)](
        bmm_34,
        add_44,
        random_values,
        out_base,
        K=K_LEN,
        BLOCK_K=block_k,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_attention_softmax_dropout(
    bmm_34: torch.Tensor,
    add_44: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    out_base: torch.Tensor | None = None,
    random_values: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 8,
) -> torch.Tensor:
    _check_shape_params(_shape_param_0, _shape_param_1, _shape_param_2)

    if random_values is None:
        random_values = _inductor_random_like_repro(inductor_seeds)
    if out_base is None:
        out_base = torch.empty_like(bmm_34)
    return _launch_oracle(
        bmm_34,
        add_44,
        random_values,
        out_base,
        block_k=block_k,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([64, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([64], i64), S([8, 8, 1024, 1024]), S([8, 8, 1024, 1024]), S([64, 1024, 1024]))")
def oracle_forward(inputs):
    return oracle_full_attention_softmax_dropout(*inputs)


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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
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
