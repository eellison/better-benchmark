# Inductor Writeup: Irregular Gather Reduce

## Status
- Queue id: `irregular_gather_reduce`
- Priority: P2 / deferred
- Oracle: `repros/canonical/sum_adeaebad93f7/oracle_fused_gather_reduce.py`

## Target
`sum_adeaebad93f7`, isolated Demucs augmentation gather/sort/cat/reduce pattern.

## Plan
Only implement if this isolated case becomes important. Pattern-rewrite chained `gather/cat/gather/view/gather -> sum(dim=1)` into a fused gather-reduce over broadcasted low-rank index structure. Keep tiny RNG/sort/index generation separate.

## Hooks
FX graph pattern around gather/cat/view/gather/reduce; possible lowering to a custom Triton gather-reduce template.

## Validation
Compare against oracle script and final forced coordinate-descent configs. Do not use memcopy SOL as the only floor because irregular source/intermediate traffic dominates.
