# pointwise_c14f03aac63b - Multi-Output Causal Mask (5.13x -> 0.71x FIXED)


## Measured Timings
- Oracle: 7.39 us
- Compile (CDT): 5.66 us
- Ratio: 0.77x

## Classification
ALGEBRAIC_REWRITE (FX pass: graph-output deduplication)

## Root Cause
The repro constructs 32 identical causal masks (fp16 [1,1,512,512]) via 32 separate
`aten.where.self` calls on a shared `le` predicate. Each `where` node uses different
`full` nodes as true/false values, but those `full` nodes produce identical scalars
(0.0 and -inf). Inductor previously scheduled these as 32 separate computations.

## Fix Implemented
Added `dedupe_graph_outputs` FX pass that performs structural CSE on graph outputs:
1. Computes a structural hash for each graph output node (based on op, target,
   and recursive hashes of arguments)
2. Groups outputs with matching hashes
3. Confirms structural equality via recursive comparison
4. Replaces all duplicates with the canonical (first) node
5. DCE removes dead subgraphs

After dedup, the graph has ONE `where` node referenced 32 times in the output tuple.
Inductor generates a single kernel computing into one buffer, returned 32 times.

## Commit
- Hash: 30cc64e2343 on branch pr-184905
- Files:
  - `/tmp/pytorch-work/torch/_inductor/fx_passes/dedupe_graph_outputs.py` (new)
  - `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (wiring)
  - `/tmp/pytorch-work/torch/_inductor/config.py` (config flag)

## Kernel Count
- Before: 32 separate where kernels (or 1 combo kernel computing 32x)
- After: 1 kernel, 1 buffer, returned 32 times (aliased outputs)
- Oracle: 1 kernel, 32 separate output buffers (stores 32x)

## Performance Results
| Config | Compile (us) | Oracle (us) | Ratio |
|--------|-------------|-------------|-------|
| Before fix | ~38.4 | 7.49 | 5.13x |
| After fix (dedupe_graph_outputs) | 5.25 | 7.42 | 0.71x |

The fix is BETTER than the oracle because it aliases all 32 outputs to one buffer
(zero extra memory traffic), while the oracle writes to 32 separate buffers.

## Config Gate
`torch._inductor.config.dedupe_graph_outputs = True` (default enabled)
Env var: `TORCHINDUCTOR_DEDUPE_GRAPH_OUTPUTS=0` to disable.

## File References
- `torch/_inductor/fx_passes/dedupe_graph_outputs.py` - the dedup pass
- `torch/_inductor/fx_passes/post_grad.py:687-694` - pass registration
- `torch/_inductor/config.py:1249-1253` - config flag

## Bugfix 2026-06-10: crash on AOT backward graphs (wave-0c finding)

The pass crashed on resnet18's BACKWARD graph under autocast bf16 with
"Argument 'sum_8' of Node 'mul_5' was used before it has been defined"
(same class as the timm_repvgg_a2_train sweep failure, "sum_4 used before
defined").

Root cause (corrected from the wave-0c "AOT hands non-topo graphs" diagnosis):
the input graph IS topologically sorted at pass entry (post_grad runs
stable_topological_sort immediately before). The bug was canonical-node
selection: the pass took the FIRST member of each duplicate group in
OUTPUT-TUPLE order as canonical. In AOT backward graphs a duplicate output
(BN bias-grad `sum`) can also have a non-output user (`mul` in the
input-grad chain) defined earlier in graph order than the output-tuple-first
node. `replace_all_uses_with` then rewired that early user to a LATER node,
breaking topo order and failing `graph.lint()` after DCE.

Fix: within each structurally-equal pair, keep whichever node is earlier in
graph order. Since the graph is topo-sorted, every user of the later
duplicate appears after the earlier canonical.

- Commit: bb9bac0cd02 on pr-184905
- Verified: resnet18 timm train (autocast bf16) compiles, 2 steps run, pass
  fires 3x; timm_repvgg_a2_train full_graph_001 compiles + runs, pass fires
  39x; this repro's oracle bench unchanged (0.678x, no regression).
