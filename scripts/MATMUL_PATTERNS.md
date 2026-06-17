# Matmul partitioning views (epilogues, mm→pw→mm chains, fan-in)

Three **read-only** views over the existing partitioner. The default capture
partitioner is **not changed**: these tools call the same
`capture_hook.get_fusion_partitions` (CapabilityBasedPartitioner +
`is_fusible_node`) that the capture pipeline and `model_graph_accounting` use,
then re-read the partition structure.

Why a matmul's pointwise neighbors live in a *separate* partition to begin
with: matmuls (`mm`/`addmm`/`bmm`/`_grouped_mm`/…) are **non-fusible extern
ops** (BLAS/flop-counter), so the default cut never folds them into a fusible
partition. The patterns here re-attach across that boundary.

## Patterns detected

1. **Matmul + strictly-pointwise epilogue** — a matmul whose output flows,
   through transparent view ops only, into a fusible partition whose *every*
   compute node is pointwise (bias add / activation / dtype cast / residual).
   The clean GEMM-epilogue-fusion target. `clean=True` means **every** real
   consumer of the matmul is such a partition (and it doesn't escape to a graph
   output).

2. **Matmul → pointwise → matmul (SEQUENTIAL chain)** — `M1 →
   (strictly-pointwise partition) → M2`. The MLP `linear → activation → linear`
   shape and any back-to-back GEMM with an elementwise bridge. The degenerate
   `M1 → view → M2` (no pointwise between) is counted in a **separate** bucket.

3. **Matmul fan-in: ≥2 matmuls → ONE output (CONVERGENT)** — two or more
   *independent* matmuls whose results are combined, through a pointwise
   bridge, into a single value. This is the "two matmuls reduce to one output"
   pattern: `mm(a,b) + mm(c,d)` (parallel projection / residual) or
   `silu(mm(x,Wg)) * mm(x,Wu)` (gated / SwiGLU MLP). Distinct from pattern 2:
   there the matmuls are *stacked* (one feeds the next); here they are
   *siblings* that merge. Reported with the combine op (`add`/`mul`/…), the
   arity, the bridge ops, and whether the bridge maps to a corpus repro.

"Strictly pointwise" = partition has ≥1 compute node and all of them are
pointwise. A reduction (softmax / layernorm / sum / mean) **disqualifies** the
partition — that's a different, harder fusion target.

Pointwise classification uses `torch.Tag.pointwise` (reliable on the post-grad
tensor overloads: `add.Tensor`, `mul.Tensor`, `where.self`, activations,
`prims.convert_element_type`) plus `aten._to_copy` added explicitly (the dtype
cast in bf16 GEMM epilogues is untagged).

## Run

```bash
# One model -> JSON summary on stdout
python scripts/matmul_pattern_analysis.py --model BertForMaskedLM

# Whole corpus -> JSON + markdown under investigation_results/matmul_patterns/
python scripts/matmul_pattern_analysis.py --all \
    --output-dir investigation_results/matmul_patterns

# Worked example + regression tests (CPU only, no GPU/timing)
python scripts/test_matmul_patterns.py
```

`--device` defaults to `cuda` for input construction but tracing is fake-mode;
pass `CUDA_VISIBLE_DEVICES=""` to run purely on CPU (no GPU contention with a
running bench fleet).

## Output

`investigation_results/matmul_patterns/` — one **deduped** JSON per pattern
(each collapses to UNIQUE op-structure signatures, shapes ignored, like the
corpus `pattern_hash`; `n_occurrences` + `occurrences_by_model` + one concrete
`example` per signature):

- `index.json` — file manifest + unique-vs-total counts.
- `summary.json` — corpus rollup (counts + breakdowns).
- `epilogues.json` — unique matmul + strictly-pointwise epilogue signatures.
- `chains.json` — unique `mm → pointwise → mm` sequential-chain signatures.
- `fanins.json` — unique `≥2 matmuls → 1 output` signatures, **clean
  memory-eliminating first** (strictly-pointwise bridge + `all_exclusive`),
  then by `total_eliminable_read_bytes`.
- `failures.json` — graphs that failed to load.
- `SUMMARY.md` — human-readable totals + top signatures + examples.

Pass `--combined-json` to also emit the single nested `matmul_patterns.json`
(the full un-deduped per-graph dump; ~15 MB, off by default).

## Files

- `scripts/matmul_pattern_analysis.py` — detector + CLI.
- `scripts/test_matmul_patterns.py` — test file + worked example.
