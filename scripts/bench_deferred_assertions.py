"""
Measure deferred assertions overhead across indirect-indexed repros.

Compares 3 configs:
  1. deferred (new default): assert_indirect_indexing=True, deferred_indirect_assertions=True
  2. eager assertions: assert_indirect_indexing=True, deferred_indirect_assertions=False
  3. no assertions: assert_indirect_indexing=False

Reports overhead % relative to no-assertions baseline.

Uses fresh inductor caches per config to ensure autotuning is independent.
"""
import sys
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')

import torch
import torch._dynamo
import torch._inductor.config as cfg
import importlib.util
import math
from triton.testing import do_bench
from torch._inductor.utils import fresh_inductor_cache

REPROS = [
    ('sum_sum_mean_9af96955f8cc', 'Qwen3 RMSNorm (index_put)'),
    ('sum_sum_mean_1dfbbe76c078', 'Qwen3 RMSNorm'),
    ('sum_sum_sum_3348817ad678', 'Qwen3 MoE router'),
    ('amax_sum_sum_dc96a87ba8db', 'Qwen softmax'),
    ('amax_sum_5f0c26b7e967', 'T5 attn + embedding gather'),
    ('sum_403da67893be', 'CE backward (one-hot gather)'),
]


def load_repro(repro_id):
    path = f'/tmp/scratch_space/better_benchmark/repros/canonical/{repro_id}/repro.py'
    spec = importlib.util.spec_from_file_location(f'repro_{repro_id}', path)
    mod = importlib.util.module_from_spec(spec)
    mod.device = torch.device
    mod.inf = math.inf
    mod.nan = math.nan
    spec.loader.exec_module(mod)
    return mod


def bench_config(mod, assert_indirect, deferred):
    """Compile and benchmark with given assertion config (fresh cache)."""
    with fresh_inductor_cache():
        cfg.coordinate_descent_tuning = True
        cfg.assert_indirect_indexing = assert_indirect
        cfg.deferred_indirect_assertions = deferred

        torch._dynamo.reset()

        instance = mod.Repro()
        make_fn = mod.make_inputs if hasattr(mod, 'make_inputs') else mod._default_make_inputs
        inputs = make_fn()

        compiled = torch.compile(instance)
        with torch.no_grad():
            for _ in range(5):
                compiled(*inputs)

        # Use microseconds
        t = do_bench(lambda: compiled(*inputs), warmup=50, rep=200, return_mode='min') * 1000
        return t


def main():
    print("=" * 100)
    print("Deferred Assertions Impact: Indirect-Indexed Repros")
    print("=" * 100)
    print()
    print(f"{'Repro':<36} {'Label':<28} {'Deferred':>10} {'Eager':>10} {'No-Assert':>10} {'Def OH%':>8} {'Eager OH%':>9}")
    print("-" * 100)

    results = []
    for repro_id, label in REPROS:
        try:
            mod = load_repro(repro_id)

            # 1. Deferred assertions (new default)
            t_deferred = bench_config(mod, assert_indirect=True, deferred=True)

            # 2. Eager assertions (old behavior)
            t_eager = bench_config(mod, assert_indirect=True, deferred=False)

            # 3. No assertions (baseline)
            t_none = bench_config(mod, assert_indirect=False, deferred=False)

            oh_deferred = (t_deferred / t_none - 1) * 100
            oh_eager = (t_eager / t_none - 1) * 100

            print(f"{repro_id:<36} {label:<28} {t_deferred:>8.1f}us {t_eager:>8.1f}us {t_none:>8.1f}us {oh_deferred:>+7.1f}% {oh_eager:>+8.1f}%")
            results.append({
                'repro_id': repro_id,
                'label': label,
                't_deferred': t_deferred,
                't_eager': t_eager,
                't_none': t_none,
                'oh_deferred_pct': oh_deferred,
                'oh_eager_pct': oh_eager,
            })
        except Exception as e:
            print(f"{repro_id:<36} {label:<28} ERROR: {e}")

    print("-" * 100)
    if results:
        avg_def = sum(r['oh_deferred_pct'] for r in results) / len(results)
        avg_eager = sum(r['oh_eager_pct'] for r in results) / len(results)
        print(f"{'AVERAGE (all)':<66} {'':>10} {'':>10} {avg_def:>+7.1f}% {avg_eager:>+8.1f}%")
        non_outlier = [r for r in results if r['oh_deferred_pct'] < 20]
        if non_outlier and len(non_outlier) < len(results):
            avg_def_no = sum(r['oh_deferred_pct'] for r in non_outlier) / len(non_outlier)
            avg_eager_no = sum(r['oh_eager_pct'] for r in non_outlier) / len(non_outlier)
            print(f"{'AVERAGE (excl outliers >20%)':<66} {'':>10} {'':>10} {avg_def_no:>+7.1f}% {avg_eager_no:>+8.1f}%")
    print()
    print("Legend: Def OH% = deferred vs no-assert, Eager OH% = eager vs no-assert")
    print("Positive = slower than no-assert baseline")
    print()
    print("Notes:")
    print("- Each config uses a fresh inductor cache (independent autotuning)")
    print("- Deferred moves tl.device_assert after reduction loops (reduces branch divergence)")
    print("- Both deferred and eager add _all_valid tracking + tl.where clamp (register pressure)")


if __name__ == '__main__':
    main()
