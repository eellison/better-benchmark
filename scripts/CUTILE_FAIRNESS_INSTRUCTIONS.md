# cuTile fairness-fix instructions

**Goal: compiler-vs-compiler comparison.** The Triton and cuTile kernels
must do the same work at the kernel level. When Triton computes `sum`,
`mean`, `var_mean`, `erf`, `softmax`, or `log_softmax` INSIDE its kernel,
the cuTile port must also compute these INSIDE its kernel (using `ct.sum`,
`ct.exp`, hand-computed erf polynomial, etc.), NOT push them to torch ops
in `oracle_forward`.

Currently, some cuTile ports have `torch.sum(...)`, `.mean(dim=...)`,
`.var_mean(...)`, `torch.special.erf(...)`, `torch.softmax(...)` etc. in
`oracle_forward` when the corresponding Triton oracle keeps those ops
inside the kernel. This gives cuTile an unfair advantage (torch may be
faster than a naive cuTile implementation) AND masks the true compiler
comparison.

## Read first

1. `/home/dev/better-benchmark/scripts/cutile_reference.md`
2. `/home/dev/better-benchmark/scripts/CUTILE_FIX_INSTRUCTIONS.md`

## Rewriting procedure per oracle

1. Read `/home/dev/better-benchmark/repros_cutile/canonical/<name>/oracle.py` (current).
2. Read `/home/dev/better-benchmark/repros/canonical/<name>/oracle.py` (Triton reference).
3. Compare: which reductions / math ops does the Triton kernel do inside its
   `@triton.jit` body? Every such op should be inside the cuTile `@ct.kernel`
   body too.
4. Rewrite the port so all in-kernel computation happens inside `@ct.kernel`
   using cuTile primitives:
   - `torch.sum(x, dim=...)` → `ct.sum(x, axis=...)`
   - `torch.mean(x, dim=...)` → `ct.sum(x, axis=...) * (1.0 / N)`
   - `torch.var_mean(x, dim=...)` → compute mean via `ct.sum`, then variance via `ct.sum((x-mean)**2)`
   - `torch.special.erf(x)` → Abramowitz-Stegun 7.1.26 polynomial in-kernel (see other rescued oracles)
   - `torch.softmax(x, dim=-1)` → `ct.max`/`ct.exp`/`ct.sum` in-kernel
5. `oracle_forward` should be lean: allocations, view/reshape/permute
   (metadata-only, no `.contiguous()` on already-contig data), maybe RNG
   pre-gen via `torch.ops.prims.inductor_random.default` when the Triton
   oracle's eager-fallback path uses that same trick.
6. Validate: `python /home/dev/better-benchmark/scripts/validate_cutile_oracle.py /home/dev/better-benchmark/repros_cutile/canonical/<name>`.
7. If numerics still pass, done. If numerics fail past tolerance after a
   couple of tries, keep the previous cuTile port (revert your edit).

## Fairness constraints (do NOT violate)

- **Do NOT increase fusion** beyond what Triton already does.
- **Do NOT change the number of kernel launches** unless the current port
  has extra torch-only launches that a proper cuTile kernel would eliminate.
- **Match Triton's BLOCK sizes** (BLOCK_M, BLOCK_H, etc. from `@oracle_impl`
  kwargs).
- **Match Triton's number of kernels**: 1 `@triton.jit` → 1 `@ct.kernel`.

## Report format

```json
{"batch": "fairness_batch_XXX.txt", "fixed": N, "reverted": N, "no_change": N,
 "results": {"<name>": "fixed" | "reverted" | "no_change:<reason>", ...}}
```

## Time budget

~5-8 min per oracle. Move on if stuck.
