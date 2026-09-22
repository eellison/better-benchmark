# cuTile RESCUE port instructions

An earlier pass converted 1727 Triton oracles; ~1300 were stubbed with
`NotImplementedError`. Many of those stubs are UNNECESSARY — the port is
actually possible. Your job is to revisit a batch of stubs and port the ones
that can be ported.

## Read first (in this order)

1. `/home/dev/better-benchmark/scripts/CUTILE_AGENT_INSTRUCTIONS.md` — original rules.
2. `/home/dev/better-benchmark/scripts/cutile_reference.md` — Triton→cuTile cheat sheet.
3. This document — rescue-specific guidance below.

## Portable features that were incorrectly stubbed

These stubs are USUALLY portable; port them if the Triton oracle uses these:

### 1. `tl.inline_asm_elementwise` for exact RN rounding
Triton often uses inline PTX for `add.rn.f32`, `mul.rn.f32`, `fma.rn.f32`,
`cvt.rn.bf16.f32`, `sub.rn.f32`, `div.rn.f32`. These enforce IEEE 754
round-to-nearest-even. **cuTile's default arithmetic on f32/bf16 is ALREADY
round-to-nearest-even.** So `_f32_add(a, b)` in Triton is just `a + b` in
cuTile; `x.to(tl.bfloat16, fp_downcast_rounding="rtne")` is just
`ct.astype(x, ct.bfloat16)`. Do NOT stub for this; port it.

### 2. Non-power-of-2 shapes with masked stores
cuTile requires tile shapes to be powers of 2. But you can:
- Load with `padding_mode=ct.PaddingMode.ZERO` (out-of-bounds → 0)
- Compute on the rounded-up tile
- For stores where OOB elements would be undefined, use one of:
  - A tile size that exactly divides the array (grid `cdiv(N, BLOCK)`).
  - Store to a padded/rounded temp buffer, then `torch.narrow` / copy the
    valid region using a torch op after the kernel.
  - `ct.scatter` with a mask (only writes elements where mask is True).

For reductions where the tail is masked-out, `ct.where(col_mask, x, 0.0)`
before the reduction gives the correct sum (0 doesn't perturb sum/max).
For softmax normalization, `ct.where(col_mask, x, -inf)` for max, then
`ct.where(col_mask, exp(...), 0.0)` for sum.

### 3. `ct.gather` for embedding lookups / advanced indexing
cuTile has `ct.gather(array, indices)`. For a 1D array and 1D int index
tile: `values = ct.gather(x, idx)`. See the docs — works well for
embedding tables.

### 4. `ct.atomic_add` / `ct.scatter` for scatter-reduce
cuTile has full atomic ops. Use `ct.atomic_add(dst, index=(...), tile=val)`.
Portable.

### 5. Multi-kernel pipelines
If Triton uses N sequential kernels, just launch N cuTile kernels sequentially
from `oracle_forward`. Fine.

### 6. Missing `ct.erf`
Compute erf using pytorch's `torch.special.erf` on a temporary tensor
(outside the kernel), then load it inside the kernel. Or implement via
`2*Phi(x*sqrt(2)) - 1` with `Phi = 0.5*(1 + tanh(sqrt(2/pi)*(x+0.044715*x**3)))`
if you need a fast approximation. Portable.

## Truly unportable — LEAVE as stubs

- **`tl.rand` / `torch.ops.prims.inductor_random`**: cuTile has no
  on-device RNG. The reference implementation depends on Inductor's
  Philox stream from a seed tensor.
  - Exception: if the RNG output can be pre-generated via
    `torch.ops.prims.inductor_random` and passed in as an input, it may
    be portable. Check whether the Repro passes the random tensor
    directly.
- **Truly complex BN-training with running-stat mutation via `copy_`**:
  If the oracle needs to mutate a running-mean/running-var tensor via
  `torch.ops.aten.copy_.default` inside a CUDA-graph-captured region,
  and cuTile can't express this in one kernel, leave it stubbed.

## Your workflow

1. Read `scripts/cutile_batches/rescue_batch_XXX.txt` — one canonical name per line.
2. For each name:
   - Read `repros_cutile/canonical/<name>/oracle.py` — the current stub.
   - Read the ORIGINAL `repros/canonical/<name>/oracle.py` — the Triton oracle.
   - Read `repros/canonical/<name>/repro.py` — the reference module.
   - Decide: is this actually portable per the categories above?
   - If yes: REPLACE the stub with a real cuTile port.
   - If no: leave it (existing stub is fine).
3. Validate: `python scripts/validate_cutile_oracle.py repros_cutile/canonical/<name>`.
4. If it passes numerics: mark done. If it fails after a couple of tries,
   revert to the stub.

## Report format at end

```json
{"batch": "rescue_batch_XXX.txt",
 "rescued": N,  // stubs replaced with working ports
 "kept_as_stub": N,  // stubs that were legitimately unportable
 "failed": N,  // attempted rescue but numerics failed → left as stub
 "results": {"<name>": "rescued" | "kept_as_stub" | "failed:<reason>", ...}}
```

## Time budget

Aim for ~90 minutes wall-clock. Don't spend more than ~5 minutes per port.
