# pointwise_c3360ba4828a - Virtual Cat Dropout Masks (SqueezeNet)

## Benchmark Result
- Oracle: 149.44 us
- Compile: 203.62 us
- Ratio: 1.362x
- Status: GOOD (oracle wins)

## Classification
SCHEDULER_FUSION

## Root Cause

The repro is from SqueezeNet training (`torchbench_squeezenet1_1_train`). It computes:
1. ReLU on two [512,256,13,13] inputs
2. Channel-dimension cat -> [512,512,13,13]
3. Inductor seeded dropout (p=0.5) on the concatenated result
4. Two boolean backward masks (le_scalar on the ReLU outputs)

### Oracle approach (1 kernel):
A single Triton kernel iterates over the full [512,512,13,13] domain and:
- Determines which input half it belongs to (channel < 256 or >= 256)
- Reads the appropriate input, computes ReLU inline
- Uses the full output element index for `tl.rand(seed, offset)` to compute dropout
- Stores dropout-scaled result + both boolean masks
- Total: 1 kernel, reads each input once, writes 3 outputs

### Inductor approach (2 kernels):
1. Kernel 1 (combo or run-twice): Computes ReLU on each input, writes into cat buffer slices, writes boolean masks
2. Kernel 2: Reads full [512,512,13,13] cat buffer, generates random, applies dropout mask, writes output

The bottleneck: kernel 2 re-reads 44M elements (512*512*13*13*4 = ~170MB) that kernel 1 just wrote.

## Kernel Count
- **Oracle: 1 kernel**
- **Inductor: 2 kernels**

## Why Inductor Cannot Do This Today

There are TWO barriers that BOTH prevent single-kernel execution:

### Barrier 1: `result.realize()` in inductor_random (lowering.py:3021)

The `inductor_random` lowering unconditionally calls `result.realize()`. This forces the random output into a separate ComputedBuffer. However, removing this alone does NOT fix the problem — the random gets fused with its downstream consumers (gt, mul) into kernel 2, but kernel 1 (relu+cat writes) and kernel 2 (random+dropout) still remain separate.

Investigation shows that `result.realize()` is actually unnecessary for correctness: the random uses `FixedLayout.make_indexer()` with contiguous strides, so `random_pos(index)` computes deterministic offsets regardless of iteration order. The realize exists as a conservative guard for RNG reproducibility that is overly broad.

### Barrier 2: Domain mismatch between cat writers and cat consumer

The fundamental blocker is that:
- Cat writers iterate over [512,256,13,13] = 22M elements (one per input half)
- Cat consumer (dropout) iterates over [512,512,13,13] = 44M elements (full cat)

The scheduler cannot fuse kernels with different iteration domains. The cat is a NopKernel (virtual layout) — it doesn't compute anything, it just assigns storage offsets. But the relu writers MUST write to the cat buffer at specific offsets (one writes to the first 256 channels, the other to channels 256-511). The dropout then reads the full 512-channel buffer.

For single-kernel execution, the kernel would need to:
- Iterate over [512,512,13,13]
- For each element, determine which input it came from (channel < 256?)
- Read from the appropriate input
- Compute ReLU inline
- Generate random and apply dropout
- Write output + both boolean masks

This requires "reading through the virtual cat" — a capability the scheduler does not have.

### What would need to change

1. **For the inductor_random realize**: Remove `result.realize()` at lowering.py:3021, or make it conditional on whether downstream consumers need deterministic layout. This alone doesn't fix the 2-kernel problem but is a prerequisite.

2. **For the domain mismatch**: The scheduler needs a new fusion mode: "inline-through-virtual-concat." When a NopKernel (cat) produces a buffer that is consumed by a pointwise op, and the cat's inputs are cheap pointwise ops, the scheduler should:
   - Iterate over the cat's output domain
   - For each element, determine which cat input it belongs to (using offset arithmetic)
   - Inline the corresponding input's computation
   - Fuse with the downstream consumer

   This is architecturally similar to the `inline_recomputable_producers` pass (see pointwise_e26de0a669ae writeup) but applied to virtual concat patterns.

## Config Exploration
- `combo_kernels=True`: Combines the two relu kernels into one combo kernel (still 2 total)
- `multi_kernel=2`: marginal improvement (197us)
- Removing `result.realize()`: Still 2 kernels (domain mismatch remains)

## File References
- `/tmp/pytorch-work/torch/_inductor/lowering.py:3021` — the `result.realize()` call
- `/tmp/pytorch-work/torch/_inductor/ir.py:6462` — ConcatKernel (NopKernel) definition
- `/tmp/pytorch-work/torch/_inductor/ir.py:6559` — `realize_into` forces relu inputs to materialize into cat buffer
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` — fusion scoring and domain matching

## Relationship to pointwise_e26de0a669ae

Both repros share the same fundamental pattern: a cheap producer's computation should be inlined into a consumer that has a DIFFERENT iteration domain. The architectural fix (`inline_recomputable_producers`) applies to both:

| Aspect | e26de0a669ae (stencil) | c3360ba4828a (concat) |
|--------|----------------------|---------------------|
| Producer | BN+ReLU [128,192,71,71] | ReLU [512,256,13,13] x2 |
| Consumer | MaxPool [128,192,35,35] | Dropout [512,512,13,13] |
| Domain change | 4:1 reduction (stride-2) | 1:2 expansion (cat) |
| Access pattern | 9 shifted reads (stencil) | Conditional read (channel offset) |
| Key mechanism | Recompute BN at each stencil pos | Inline relu through virtual cat |

## Status
Design doc only — fix not implemented. Requires both the `inductor_random` realize removal AND a "read-through-virtual-concat" mechanism in the scheduler.
