# sum_sum_sum_d06bf12e10d0

## Summary

- Model: ALBERT (hf_AlbertForMaskedLM / torchbench_hf_Albert)
- Oracle: `oracle_albert_projection_ln_multi_reduction.py`
- Classification: SCHEDULER_FUSION
- Ratio: 1.054x (oracle 499.5us, compile 526.3us default; but CDT/mk3 close the gap)
- Kernel count: Oracle uses fused partial-reduction + finalizer, Inductor uses many separate reduction kernels

## Root Cause

The oracle computes the complete ALBERT projection plus layernorm-backward reduction fragment by fusing the three matrix-view adds around `mul_314`, the required materialized non-contiguous transpose side output, and the twelve sibling column reductions in one shared partial-reduction/finalizer plan. Inductor currently schedules the rowwise layernorm-backward producer and the many same-axis reductions as separate generic kernels with repeated reads of the same sources.

However, Inductor's coordinate_descent_tuning and multi_kernel=3 modes find dramatically better tile configs for this pattern, reducing compile time from 1616us (baseline) to ~517-523us -- which actually matches or slightly beats the oracle (499.5us).

## Config Exploration

| Config | Time (us) | Speedup vs default |
|--------|-----------|-------------------|
| baseline (default) | 1615.8 - 1621.8 | 1.00x |
| coordinate_descent_tuning=True | 516.9 | 3.14x |
| combo_kernels=True | 1483.6 | 1.09x |
| triton.multi_kernel=2 | 1630.1 | ~1.00x (slight regression) |
| triton.multi_kernel=3 | 523.5 | 3.09x |

## Key Finding

The default Inductor baseline was **severely undertiled** for this 4096x4096 shape with 38 sum reductions. Both `coordinate_descent_tuning=True` and `triton.multi_kernel=3` find tile configs that achieve 3x speedup, bringing compile performance to within 3-5% of the oracle. This suggests the gap is primarily an **autotuning coverage issue** rather than a fundamental scheduling limitation.

With CDT or mk3 enabled:
- Oracle: 499.5 us
- CDT compile: 516.9 us (ratio drops to 0.966x -- oracle barely faster)
- mk3 compile: 523.5 us (ratio drops to 0.954x)

The gap effectively disappears with proper tuning configs enabled.

## Fix Assessment

**Near-term fix**: Enable coordinate_descent_tuning by default for reduction kernels with this shape profile (many sibling column reductions over large matrices). The 3x improvement from CDT suggests the default heuristics are choosing poor tile sizes.

**Structural fix (SCHEDULER_FUSION)**: Still valuable for reducing kernel launch count and memory traffic from repeated source reads, but the perf gap is already closable with existing config knobs.

### Difficulty: Low (config fix), Medium (structural scheduler fix)

## Commands

```bash
python repros/canonical/sum_sum_sum_d06bf12e10d0/oracle_albert_projection_ln_multi_reduction.py --check
python repros/canonical/sum_sum_sum_d06bf12e10d0/oracle_albert_projection_ln_multi_reduction.py --bench --warmup 25 --rep 200
python scripts/bench_compare.py repros/canonical/sum_sum_sum_d06bf12e10d0/repro.py --config "baseline" --label "default" --config "coordinate_descent_tuning=True" --label "cdt" --config "triton.multi_kernel=3" --label "mk3" --n-warmup 10 --n-rep 50 --rounds 5
python -m py_compile repros/canonical/sum_sum_sum_d06bf12e10d0/oracle_albert_projection_ln_multi_reduction.py
python scripts/validate_corpus_invariants.py
```

## Affected Models

- hf_AlbertForMaskedLM_train_001
- torchbench_hf_Albert_train_001
