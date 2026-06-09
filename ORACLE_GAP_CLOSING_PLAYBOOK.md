# Oracle Gap Closing Playbook

Standard methodology for iterating on oracle benchmarks to close or confirm
the gap between oracle performance and `torch.compile`.

## When to use this playbook

After processing an oracle batch (measuring `oracle_us` vs `compile_us`), any
repro with `ratio > 1.1` (REAL_GAP) is a candidate for iteration. The goal is
either:

1. **Close the gap** -- make the compile path faster (proving the gap was a
   config/tuning issue, not a fundamental Inductor limitation)
2. **Confirm the gap** -- demonstrate that no available Inductor config can
   match the oracle, confirming the gap is a real missing optimization

## Step 1: Characterize the compile path

```bash
# What kernels does Inductor generate?
python3 -c "
import torch, torch._dynamo, torch._inductor.config as cfg, glob, os
from torch._inductor.utils import fresh_inductor_cache
from torch._inductor.codecache import cache_dir
# ... load repro ...
torch._dynamo.reset()
cfg.coordinate_descent_tuning = True
with fresh_inductor_cache():
    compiled = torch.compile(instance)
    compiled(*inputs)
    torch.cuda.synchronize()
    # find generated wrapper, count .run() calls
"
```

Key questions:
- How many kernel launches? (Multi-launch = fusion opportunity)
- What's the reduction strategy? (persistent vs non-persistent)
- Is the generated kernel autotuned or using heuristics?

## Step 2: Config sweep

Try these Inductor configs from most to least likely to help:

| Config | When it helps |
|--------|--------------|
| `coordinate_descent_tuning=True` | Always try first; tunes block sizes |
| `max_autotune=True` | Tries more configs; helps reductions |
| `combo_kernels=True` | Horizontal fusion of small kernels |
| `force_persistent_reductions=True` | When reduction is small enough for L2 |
| `aggressive_fusion=True` | More liberal fusion decisions |
| `triton.multi_kernel=3` | Multi-kernel strategy selection |

Template for sweep:
```python
configs = [
    ('default', {}),
    ('CD', {'coordinate_descent_tuning': True}),
    ('max_autotune+CD', {'max_autotune': True, 'coordinate_descent_tuning': True}),
    ('CD+combo', {'coordinate_descent_tuning': True, 'combo_kernels': True,
                  'combo_kernel_per_subkernel_blocks': True}),
    ('CD+persistent', {'coordinate_descent_tuning': True,
                       'force_persistent_reductions': True}),
]
for label, config in configs:
    torch._dynamo.reset()
    with fresh_inductor_cache():
        for k, v in config.items():
            setattr(cfg, k, v)
        compiled = torch.compile(instance)
        # ... bench ...
```

## Step 3: Algebraic elimination (manual rewrite test)

If the oracle exploits dead-code elimination (e.g., only one row is used from
a full normalization), test whether manually rewriting the repro module helps:

```python
class ReproOptimized(torch.nn.Module):
    def forward(self, ...):
        # Only compute the live rows/outputs
        ...
```

If the rewritten module compiles to match the oracle, the gap is CONFIRMED as
an algebraic-elimination opportunity in Inductor's graph simplifier.

## Step 4: Oracle kernel tuning

If the compile path cannot be improved, check if the oracle itself is optimal:

- Try different `num_warps` (1, 2, 4, 8)
- Try different `BLOCK` sizes
- For row-parallel kernels: verify occupancy vs. SM count
- For column reductions: check if `BLOCK_M` covers the full reduction

## Step 5: Classify the outcome

After iteration:

| Outcome | CSV status | Meaning |
|---------|-----------|---------|
| Compile matches oracle (ratio < 1.1) | `closed` | Gap was a tuning issue |
| Compile improves but gap remains | `needs_work` | Note best config in CSV |
| No config helps; gap confirmed | `needs_work` | True Inductor limitation |
| Oracle is actually slower | `closed` | BAD_ORACLE; Inductor optimal |

## Step 6: Update the CSV

```
repro_id,diagnosis,impl_status,notes
var_mean_XYZ,ALGEBRAIC_ELIMINATION,needs_work,fresh_YYYY-MM-DD: compile=X.Xus oracle=Y.Yus gap=Z.Zx; <model> <description>; best_config=max_autotune; true_floor=yes
```

## Common gap patterns and expected outcomes

### ALGEBRAIC_ELIMINATION (dead row/output elimination)
- Typical gap: 2-4x (computing N rows when only 1 is used)
- Config sweep rarely helps (Inductor doesn't know rows are dead)
- Manual rewrite test confirms the opportunity
- Resolution: Inductor graph pass needed

### SCHEDULER_FUSION (multi-kernel -> single kernel)
- Typical gap: 2-3x (multiple launches vs one)
- `combo_kernels=True` sometimes helps
- If Inductor already fuses, gap is codegen quality
- Resolution: fusion pattern recognition in scheduler

### NEW_PATTERN (dedicated template)
- Typical gap: 1.3-2x (generic template vs specialized)
- `max_autotune` sometimes narrows the gap
- Gap persists because Inductor uses generic reduction machinery
- Resolution: add a specialized template/lowering

### BANDWIDTH_BOUND (at memory floor)
- Typical gap: < 1.15x or BAD_ORACLE
- Both oracle and compile are near memcpy SOL
- Usually classified as `closed` or `AT_FLOOR`

## Reporting format

After iteration, update the batch results JSON and CSV with:
- Best compile config found and its time
- Whether the gap is confirmed or closeable
- The recommended Inductor fix category
