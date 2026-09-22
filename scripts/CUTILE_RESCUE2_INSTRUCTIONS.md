# cuTile RESCUE #2 — nothing is unportable

The first two passes left 900+ stubs marked as unportable. On review, essentially
**all of them ARE portable**. This pass is intended to rescue the rest.

## What each category actually needs

### seeded_rng — the biggest bucket

The Triton oracle uses `tl.rand(seed, offsets)` on-device under CUDA-graph
capture. BUT every such Triton oracle already carries an eager fallback that
uses `torch.ops.prims.inductor_random.default(shape, seed, "rand")` OUTSIDE the
kernel. That's precisely the pattern cuTile should use:

1. In `oracle_forward`, call
   `random = torch.ops.prims.inductor_random.default(shape, seed, "rand")`
   ONCE, outside the kernel.
2. Pass `random` as an extra input to the cuTile kernel.
3. Inside the kernel, `ct.load(random_ptr, index=..., shape=...)` and use it
   just like the Triton `USE_RANDOM_PTR: tl.constexpr` branch does.
4. Under CUDA-graph capture the same pattern still works — `inductor_random`
   emits a graph-capturable op. If the timed path in the Triton oracle uses
   an on-device `tl.rand`, cuTile's equivalent is to pre-generate the random
   tensor as part of the graph via `torch.ops.prims.inductor_random`.

Look at the Triton oracle's `_inductor_random_for_eager_check` / `USE_RANDOM_PTR`
branch — that's exactly the code shape to use in your cuTile port.

### inline_ptx — cuTile IS already RTNE by default

`add.rn.f32`, `mul.rn.f32`, `fma.rn.f32`, `cvt.rn.bf16.f32` are all just
"round-to-nearest-even f32/bf16" — which is cuTile's default. Just use `+`,
`*`, `ct.astype(x, ct.bfloat16)` directly.

### non_pow2 (masked stores)

Load with `padding_mode=ct.PaddingMode.ZERO`. For stores, either:
- pick a `BLOCK` that divides the array, OR
- write to a padded output tensor then `torch.narrow` the valid region, OR
- use `ct.scatter(out, indices, tile)` to only write valid elements.

### multi_kernel / cooperative_split_k

Just launch multiple cuTile kernels sequentially from `oracle_forward`.
Torch reductions between kernels are fine (they're graph-capturable).

### bn_training with running-stat mutation via `copy_`

`torch.ops.aten.copy_` on a running_mean/running_var tensor works fine under
CUDA graph capture. Do the reduction in cuTile, then call `copy_` from
`oracle_forward`.

### missing_erf

Use `torch.special.erf(x)` on a temporary tensor OUTSIDE the kernel, then load
inside. Or compute `erf` via cuTile using the Abramowitz-Stegun 7.1.26
polynomial (many rescued oracles already do this — search for `_erf_approx`).

### atomic_scatter

Use `ct.atomic_add(out_ptr, index=(...), tile=val)`. cuTile has full atomic
support. For masked scatter, redirect masked-out elements to a safe throwaway
index or use `ct.where` before the atomic call.

### irregular_gather / embedding lookups

`ct.gather(table_ptr, indices_tile)` — works for 1D and 2D gather.

### torch_only — these are already working ports demoted to stubs

Just restore the oracle by writing a proper cuTile port. The prior "torch-only"
label meant the port used no cuTile kernels — that's not a legitimate cuTile
port for our comparison. You need at least ONE `@ct.kernel` doing substantive
work.

## Rules

* You MAY leave a stub if the port literally cannot be validated — but the bar
  is now: "I tried and the numerics gate rejects it beyond tolerance", not
  "I don't know how to write this".
* Every port must contain at least one `@ct.kernel` doing substantive work.
  Pure-torch oracles are NOT valid cuTile ports.
* Validate with:
  `python /home/dev/better-benchmark/scripts/validate_cutile_oracle.py <dir>`

## Report format

```
{"batch": "rescue2_batch_XXX.txt", "rescued": N, "kept_as_stub": N, "failed": N,
 "results": {"<name>": "rescued" | "kept_as_stub" | "failed:<reason>", ...}}
```
