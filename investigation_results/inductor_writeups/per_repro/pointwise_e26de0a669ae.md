# pointwise_e26de0a669ae - BN + ReLU + MaxPool + AvgPool Fusion Gap

## Benchmark Result
- Oracle: 230.43 us
- Compile: 352.19 us
- Ratio: 1.528x
- Status: GOOD (oracle wins)

## Classification
SCHEDULER_FUSION

## Root Cause

The oracle fuses BatchNorm-inference affine + ReLU + 3x3 stride-2 maxpool into **one** Triton kernel, followed by a separate avgpool kernel (2 kernels total). The critical fusion is sinking the BN+ReLU computation into the maxpool stencil so the large intermediate f32[128,192,71,71] (124M elements, ~496MB) is never materialized.

Inductor emits **three** kernels:
1. `triton_poi_fused_add_mul_reciprocal_relu_sqrt_sub_unsqueeze_0` (124M elements): BN affine + ReLU, writes full f32[128,192,71,71] to memory
2. `triton_poi_fused__low_memory_max_pool_with_offsets_..._1`: Reads the materialized BN+ReLU output back, computes 3x3 maxpool stencil, writes f32[128,192,35,35]
3. `triton_poi_fused_avg_pool2d_2`: Reads pool output, computes 3x3 stride-1 avgpool, writes final f32[128,192,35,35]

The key inefficiency is the materialization between kernels 1 and 2. The BN+ReLU output is 128*192*71*71*4 = 496 MB. Writing it and reading it back costs ~992 MB of memory traffic that the oracle avoids.

## Kernel Count
- Inductor: 3 kernels
- Oracle: 2 kernels (BN+ReLU+maxpool fused, then avgpool)

## Memory Traffic Analysis

**Oracle** (2 kernels):
- Kernel 1 (BN+ReLU+maxpool): Reads input (496 MB channels-last) + BN params (4*192*4 = 3KB), writes pool output (128*192*35*35*4 = 120 MB)
- Kernel 2 (avgpool): Reads 120 MB, writes 120 MB
- Total: ~856 MB

**Inductor** (3 kernels):
- Kernel 1: Reads input (496 MB) + params (3KB), writes BN+ReLU result (496 MB)
- Kernel 2: Reads BN+ReLU (496 MB), writes pool (120 MB)
- Kernel 3: Reads pool (120 MB), writes avg (120 MB)
- Total: ~1848 MB (2.16x more traffic)

The 1.53x performance gap is explained by the ~2.16x more memory traffic.

## Why Inductor Cannot Do This Today

The scheduler does not fuse a pointwise producer (BN+ReLU, iterating over [128,192,71,71]) with a stencil consumer (maxpool, which reads a 3x3 window of the producer's output per output element). Three independent blockers:

**Blocker 1: `realize_hint()` forces materialization at lowering time** (ir.py line 10069)
- `lowering.py:5560` calls `x.realize_hint()` before maxpool lowering
- Since BN+ReLU has `nontrivial_read_count > 1` (5 reads), it gets materialized
- Once materialized, the buffer exists as a ComputedBuffer in the graph

**Blocker 2: `score_fusion_memory` returns 0** (scheduler.py)
- The producer writes `MemoryDep('buf0', c0, {c0: 123887616})` (contiguous)
- The consumer reads `MemoryDep('buf0', 967872*c0 + 27264*c1 + 384*c2 + c3, ...)` (stencil)
- Dep set intersection is empty because index expressions differ
- `V.choices.can_fuse()` rejects with "no shared data"

**Blocker 3: `can_fuse_vertical` rejects index-mismatched deps** (scheduler.py line 8095)
- Even with `aggressive_fusion=True`, `fusable_read_and_write` can't match stencil reads to contiguous writes
- `remaining_deps & node1_buf_names` is non-empty → returns False

**Blocker 4: Different iteration domains prevent codegen fusion**
- Producer iterates over 124M elements, consumer over 30M elements
- `generate_node_schedule` requires matching `(node_numel, node_rnumel)` for same-body scheduling

## Config Exploration
- `combo_kernels=True`: no change (3 kernels)
- `aggressive_fusion=True`: bypasses score check, but can_fuse_vertical still fails
- `coordinate_descent_tuning=True`: already enabled
- `score_fusion_memory_threshold=-1`: bypasses score, but can_fuse_vertical fails

No config resolves this. The issue is architectural.

## Scheduler-Level Fix: inline_recomputable_producers (WIP)

### Approach
Added `config.inline_recomputable_producers` (default disabled) in `/tmp/pytorch-work/torch/_inductor/scheduler.py` with commit `905450a5a5d`.

The pass runs before fusion and:
1. Identifies cheap pointwise producers (broadcast-dominated: most reads are small per-channel params)
2. Checks that the producer has only one consumer (buffer can be eliminated)
3. Checks profitability (memory saved > recompute cost)
4. Rewrites the consumer's `inner_fn` to inline the producer's computation using a `WrapperHandler` that intercepts `ops.load(producer_buf, flat_idx)` and redirects to `producer_inner_fn(inverse_coords(flat_idx))`
5. Clears cached LoopBody and re-traces with the new inner_fn
6. Removes the producer from the schedule

### Results
- **Kernel count: 3 → 2** (BN+ReLU+MaxPool fused into one kernel)
- **Correctness: PARTIAL** — 49% of outputs correct (within 1e-5 of reference), 51% NaN
- The NaN is caused by the inverse indexer: `FloorDiv/ModularIndexing` decomposition of the flat stencil index produces expressions that don't simplify through Triton codegen for positions where the stencil offset crosses a stride boundary

### Root Cause of Correctness Issue
The inverse indexer converts a flat buffer index to multi-dimensional coordinates using the producer's layout strides [967872, 1, 13632, 192] (channels-last). For the first 4 stencil positions (offsets 0, 192, 384, 13632), the ModularIndexing simplifies correctly and CSE deduplicates the BN param loads. For positions 5-9 (offsets 13824, 14016, 27264, 27456, 27648), the expressions like `((13824 + x0 + 384*x1) % 13632) % 192` don't simplify to `x0` in the Triton codegen, causing the BN params to be loaded with incorrect indices.

### Fix Direction
Replace the flat-index inverse indexer with a **direct coordinate mapper** that uses the relationship between the consumer's iteration variables and the producer's coordinate space:
- Consumer iterates over [batch, channel, H_out, W_out]
- Producer coordinates are [batch, channel, 2*H_out + dh, 2*W_out + dw] where (dh, dw) are the stencil offsets
- This avoids FloorDiv/Mod decomposition entirely and produces clean index expressions

Alternatively, fix the Triton codegen's simplification of nested ModularIndexing expressions to recognize that `((a*k + b) % (a*n)) % a == b % a` when `b < a`.

## File References
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` lines 5363-5683: `inline_recomputable_producers` and helpers
- `/tmp/pytorch-work/torch/_inductor/config.py` line 920: `inline_recomputable_producers = False`
- `/tmp/pytorch-work/torch/_inductor/lowering.py` line 5560: `x.realize_hint()` in maxpool lowering
- `/tmp/pytorch-work/torch/_inductor/ir.py` line 10069: `realize_hint()` implementation

## Affected Repros
- timm_adv_inception_v3_infer, timm_inception_v3_infer
- Any CNN with BN->ReLU->Pool in inference mode
- Any model with channels-last pooling after broadcast-dominated pointwise

## Source
- Label: timm_adv_inception_v3_infer
- Pattern: channels-last BN affine -> ReLU -> 3x3 stride-2 maxpool -> 3x3 avgpool
