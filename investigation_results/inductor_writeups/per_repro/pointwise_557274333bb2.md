# pointwise_557274333bb2 — BN Affine Per-Channel Recomputation

## Summary
- **Repro**: torchbench_mnasnet1_0_infer_000
- **Oracle**: oracle_bn_affine.py
- **Ratio**: 1.399x (oracle 10.82us vs compile 15.14us)
- **Classification**: ALGEBRAIC_ELIMINATION

## Root Cause

The oracle pre-computes per-channel BN scale/shift coefficients (320 elements) once in a
small kernel, then applies a flat FMA (x * scale + shift) across the entire [256, 320, 7, 7]
output tensor. Inductor instead fuses ALL operations (convert_element_type, sub, sqrt,
reciprocal, mul, add) into a single pointwise kernel that recomputes sqrt/reciprocal for
every output element.

With shape [256, 320, 7, 7], the per-channel ops on [320] get recomputed 256 * 49 = 12,544
times each. The oracle reads each of the 4 per-channel params once per tile and does 1 FMA
per element; Inductor does sqrt + reciprocal + 2 muls + 1 sub + 1 add per element.

## Kernel Count
- **Oracle**: 2 kernels (1 small per-channel coefficient kernel + 1 FMA kernel)
- **Inductor**: 1 kernel (all ops fused, recomputing per-channel ops per-element)

## Config Exploration
- `coordinate_descent_tuning = True`: no effect on this pattern
- `combo_kernels = True`: no effect (single kernel already)
- `realize_opcount_threshold`: The per-channel chain is ~6 ops total, well below the
  default threshold of 30. Lowering this aggressively would cause materialization elsewhere.

## Why Inductor Cannot Fix This Today

The issue is that Inductor's recompute-vs-realize heuristic only considers:
1. Number of reads (4 reads, below `realize_reads_threshold=4` but barely)
2. Op count for the inner fn (6 ops, well below `realize_opcount_threshold=30`)

It does NOT consider the broadcast factor: the per-channel ops are [320]-sized but get
broadcast into [256, 320, 7, 7] (12544x amplification). The oracle recognizes that
precomputing 320 scale/shift values and reading them from a coefficient buffer saves
12544 * (sqrt + recip + muls) = massive ALU savings.

## Design Doc: Broadcast-Dominated Recomputation Heuristic

**What's needed**: The scheduler/lowering should detect when:
1. A chain of ops produces a small tensor (e.g. [C])
2. That tensor is broadcast into a much larger consumer (e.g. [N, C, H, W])
3. The broadcast factor * op_cost exceeds the cost of materializing the intermediate

**Where to fix**:
- `torch/_inductor/ir.py` — `should_realize_on_reuse()` (line ~10144): add a check for
  broadcast amplification factor. When the inner fn's numel << consumer's numel, the
  effective recomputation cost is (consumer_numel / producer_numel) * ops_per_element.
- Alternative: `torch/_inductor/fx_passes/` — add a BN affine folding pass that
  canonicalizes the sub/sqrt/reciprocal/mul/mul/add chain into a pre-computed
  scale+shift pattern before scheduling.

**Affected files**:
- `/tmp/pytorch-work/torch/_inductor/ir.py` (should_realize_on_reuse, ~line 10144)
- `/tmp/pytorch-work/torch/_inductor/config.py` (realize_reads_threshold, line 869)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (pass registration)

**Estimated affected repro count**: This pattern (BN inference affine) is extremely common
in CNN inference (ResNet, MnasNet, EfficientNet, etc.). Likely 10+ repros in the corpus.
