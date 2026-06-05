# pointwise_c14f03aac63b - Multi-Output Causal Mask (5.128x)

## Classification
SCHEDULER_FUSION

## Root Cause
The repro constructs 32 identical causal masks (fp16 [1,1,512,512]) via 32 separate
`aten.where.self` calls on a shared `le` predicate. The oracle writes all 32 outputs
from a single Triton kernel launch where each thread computes the predicate once and
stores to all 32 output buffers.

Inductor's combo_kernels mode groups these into a single kernel launch but uses
round-robin PID dispatch (`pid % 8`), meaning each thread block writes to only ONE
of the 8 grouped outputs. With the default `combo_kernel_max_num_nodes = 8`, the
kernel is launched 4 times (32/8 = 4). Even with max_num_nodes=32, each thread block
still writes to a single output buffer - the predicate is recomputed 32x total
across all thread blocks rather than computed once and stored 32 times.

## Kernel Count
- Oracle: 1 kernel, 1 launch, 32 stores per thread
- Inductor (default combo_kernels): 1 kernel, 4 launches, 8 outputs per launch
- Inductor (combo_kernel_max_num_nodes=32): 1 kernel, 1 launch, but still 32 independent branches (no multi-store)

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo_kernels=True (default max=8) | 38.4 | 4 launches of 8-way round-robin kernel |
| combo_kernel_max_num_nodes=32 | 55.92 | 1 launch but 32-way branching hurts occupancy |
| Oracle | 7.49 | Single kernel, each thread stores to all 32 outputs |

## Why Inductor Cannot Fix This Today
The combo_kernels mechanism is designed to reduce kernel launch overhead by packing
independent pointwise kernels into one grid with PID dispatch. It does NOT merge
identical computations into a single computation with multiple stores.

The correct fix is an FX-level deduplication pass:
1. Detect graph-output nodes with identical computation DAGs
2. Replace N identical nodes with 1 compute node + N output store nodes
3. Generate a single kernel body that computes once and stores N times

## File References
- `torch/_inductor/scheduler.py:3425` - combo_kernel_max_num_nodes limit
- `torch/_inductor/scheduler.py:3417` - _default_group_nodes_for_combo_kernels

## Design Doc: Graph-Output Sibling Deduplication Pass

### What's needed
An FX pass (post-grad) that identifies groups of graph-output nodes where:
- All nodes have identical computation subgraphs (same ops, same inputs)
- Nodes differ only in their output buffer

The pass would replace N identical subgraphs with 1 subgraph that writes to N
output buffers. The scheduler would then emit a single pointwise kernel with
N `tl.store()` calls.

### Affected repro count
This pattern appears in LLaVA causal mask construction and likely in other models
that duplicate attention masks across heads/layers. The 5.128x gap makes this a
high-priority fix.

### Difficulty
Medium - the FX pattern matching is straightforward, but the codegen needs to handle
multi-output pointwise kernels where outputs share the same iteration domain.
