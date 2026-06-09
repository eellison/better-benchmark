# Loop-invariant compute hoisting (design TODO #6) — implemented, measured neutral on B200

## Status: IMPLEMENTED_NEUTRAL (config-gated, default OFF)

## What was built

Generic loop-invariant code motion for looped-reduction Triton kernels, extending
the existing load-hoisting mechanism (`get_load_buffer` returning `self.body` for
r-independent direct loads) to:

1. **Pure elementwise compute ops** whose args are all loop-invariant
   (`CSEVariable in outside_loop_vars`, or sympy exprs whose free symbols are
   neither reduction-index symbols nor TMP symbols backed by in-loop vars).
   Wired via a new `Kernel.get_compute_buffer(name, args, kwargs)` hook called
   from `CSEProxy._default` (`codegen/common.py`), overridden in `TritonKernel`.
   Hoisted vars are added to `outside_loop_vars`, so chains hoist transitively
   (e.g. the full `rsqrt(var+eps)*weight` BN chain lands above the loop).
2. **Indirect loads with loop-invariant index chains** (e.g. the CE target-logit
   gather `logits[row, target[row]]`): `get_load_buffer` now returns `self.body`
   when the index's TMP symbols all resolve to outside-loop vars, there is no
   tmpmask/rindex, and no active load mask. This is the generic version of the
   CE-specific "Path B" pre-gather (9b4edfffc15 / lowering.py:9190).

Gate: `config.hoist_invariant_compute` (`TORCHINDUCTOR_HOIST_INVARIANT_COMPUTE`),
**default OFF** (see measurements).

Safety handling that survived review:
- ops with explicit CSEProxy methods (load/store/reduction/scan/sort/bucketize/
  indirect_indexing) never reach the hook, so they can't be mis-hoisted.
- `masked` bodies and r-dependent `index_expr`s fail the invariance check (graph
  bodies aren't analyzable; sympy args are checked symbol-by-symbol via
  `TritonSymbols.is_reduction_index_symbol`).
- ops that emit code directly into `kernel.compute` and return existing
  CSEVariables (frexp etc.) are not marked hoisted (`do_cse` only marks vars it
  created from string exprs).
- `_load_mask` active → no hoisting (conservative; an xmask-only load mask would
  be safe to relax later).

## Verification (codegen + correctness)

- Probe `(x * (1/sqrt(w+1e-5))[:,None]).sum(dim=1)` ([512,32768] f16): the
  `+eps → sqrt_rn → div` chain previously re-executed all 32768/R0_BLOCK
  iterations now sits above `for r0_offset ...`. assert_close PASS.
- Synthetic invariant gather `(x * scale[idx][:,None]).sum(dim=1)`: the
  `tl.load(in_ptr1 + (tmp4))` indirect load + its index chain + device_assert all
  hoisted above the loop. assert_close PASS.
- Multi-flush stress (softmax over 65536 with invariant scale, 3 sequential
  r-loops in one kernel): invariant chains hoist correctly between loops
  (`tmp37/tmp38` emitted after loop 1, before loop 2); per-iteration CSE
  invalidation preserves outside_loop_vars. assert_close PASS.
- torch.compiled cross_entropy (8192x32768): bitwise-correct vs eager.
- Corpus spot checks with flag ON: mean_7639bfb9be38 0.989 AT_FLOOR,
  var_mean_598830735cf6 1.402 (same as baseline). With flag OFF (default):
  mean_7639bfb9be38 0.976 AT_FLOOR — defaults unchanged.

## Measurements (B200, CUDAGraph, fresh cache per run): all NEUTRAL

| Workload | hoist=0 | hoist=1 |
|---|---|---|
| invariant sqrt-chain reduction [512,32768] | 5.13us | 5.15us |
| expensive invariant chain (lgamma+digamma+exp) [64,262144] | 50.38us | 51.10us |
| torch CE (online-softmax path) [8192,32768] | 175.63us | 175.67us |
| manual CE (gather + logsumexp) [8192,32768] | 236.22us | 236.27us |
| mean_7639bfb9be38 w/ rsqrt+bn-fold disabled | 1.388x | 1.351x |

## Why neutral

Looped reductions on B200 are memory-bandwidth-bound; the redundant in-loop
work is `[XBLOCK,1]`-shaped (one redundant op per program per iteration, not per
element) and fully latency-hidden behind the streaming `evict_first` loads. The
same applies to the L2-hot invariant gather. The 1.35x residual on
mean_7639bfb9be38 with band-aids disabled is a *persistent* (non-looped)
reduction — out of scope for this mechanism (no loop to hoist out of), and
already at floor via fold_bn_affine.

Implication for the band-aids: fold_bn_affine wins NOT by removing in-loop
recompute but by removing per-element loads/ops in pointwise kernels and
shrinking the op count in persistent reductions. rsqrt canonicalization wins on
instruction count in persistent/pointwise kernels for the same reason.

## When it could matter (why keep the code, default-off)

- Compute-bound or occupancy-limited looped reductions (small rnumel per block
  with expensive invariant transcendentals; low-bandwidth GPUs).
- As an enabler: hoisted scalars reduce in-loop register pressure in kernels
  where the scheduler currently spills (none identified in this corpus).
- The CE "Path B" pre-gather (lowering-level) remains the effective fix for its
  pattern; the generic codegen version subsumes it functionally but showed no
  additional win where Path B already fires.

## Commit

`/tmp/pytorch-work` branch `pr-184905`: see `[inductor] Loop-invariant compute/gather hoisting for looped reductions (neutral on B200, default off)`.
Files: torch/_inductor/codegen/common.py (get_compute_buffer hook in CSEProxy._default),
torch/_inductor/codegen/triton.py (TritonKernel.get_compute_buffer,
_is_loop_invariant_arg, mark_hoisted_compute, get_load_buffer indirect branch),
torch/_inductor/config.py (hoist_invariant_compute).
