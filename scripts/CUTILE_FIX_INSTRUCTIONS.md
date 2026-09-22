# cuTile perf-fix instructions

**Goal: compiler-vs-compiler comparison between Triton and cuTile.**

The cuTile port must match the Triton oracle's kernel structure as closely as
possible — same number of kernels, same fusion boundaries, same block sizes.
If the Triton kernel does N operations in one launch, the cuTile kernel should
do the same N operations in one `ct.launch`. The point is to measure
**compiler code-gen quality**, not to give either side unfair advantages.

## Fairness constraints

1. **Same kernel count.** If Triton uses 1 `@triton.jit`, cuTile should use 1
   `@ct.kernel`. If Triton uses 3 kernels, cuTile should use 3.
2. **Same fusion.** Every op that appears fused in one Triton kernel body
   should appear fused in the corresponding cuTile kernel body.
3. **Same block sizes.** Copy Triton's `BLOCK_M`, `BLOCK_N`, `BLOCK_H`, etc.
   verbatim into the cuTile port. If Triton uses `BLOCK=1024`, cuTile uses
   `BLOCK=1024`.
4. **No unfair torch offloading.** Do NOT push work from the kernel into
   `oracle_forward` as torch ops if Triton did that work inside the kernel.
   Only use torch for setup (allocations, shape reshaping via metadata-only
   views) or genuine wrapper ops that Triton also has to do (RNG
   pre-generation with `torch.ops.prims.inductor_random`, when Triton's
   `USE_RANDOM_PTR=True` branch does the same).
5. **No unfair fusion.** Do NOT merge multiple Triton kernels into one
   cuTile kernel to gain speed.

## Antipatterns to fix (code-quality, NOT compiler advantages)

### 1. `.contiguous()` on already-contiguous views (BIGGEST offender)

Before (wastes a full tensor copy every launch):
```python
x_2d = arg0_1.permute(0, 2, 3, 1).contiguous().view(-1, C)
```

After:
```python
x_2d = arg0_1.permute(0, 2, 3, 1).reshape(-1, C)  # metadata-only if input is channels-last
```

Verify with `x.permute(0, 2, 3, 1).reshape(-1, C).is_contiguous()`. If input
is channels-last, this is already True — the `.contiguous()` was a no-op copy.

### 2. `torch.zeros(padded_shape); padded[:H, :W] = src` — unfair pad copies

Before:
```python
x_p = torch.zeros((N, C, BLOCK_HW), device=x.device, dtype=x.dtype)
x_p[:, :, :HW] = x_c   # full-tensor copy every launch
```

After — use cuTile's `padding_mode=ct.PaddingMode.ZERO`:
```python
tile = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_HW),
               padding_mode=ct.PaddingMode.ZERO)
# tile[col >= HW] is now 0.0
```

Triton's `tl.load(..., mask=..., other=0.0)` is exactly this. cuTile's
built-in padding does the same without a torch alloc.

### 3. Tiny BLOCK sizes (this is a fairness issue too)

Look at the Triton oracle's `BLOCK` values. Match them in cuTile. Some
rescued ports use `BLOCK=1` or `BLOCK=4` when Triton uses `BLOCK=1024` — that
is BOTH poorly written cuTile AND an unfair handicap.

### 4. Extra kernel launches (fairness issue)

If Triton fuses ops A + B + C in one kernel body, and cuTile splits them
across 2 `ct.launch` calls, that's a fairness gap in cuTile's disfavor.
Merge into one cuTile kernel to mirror Triton's structure. (This is
**equalizing**, not gaining unfair fusion — Triton already fuses them.)

### 5. Torch ops replacing kernel work (fairness issue)

Some rescued cuTile ports pushed part of the computation into torch (e.g.
`torch.var_mean` instead of computing var/mean inside the kernel). If the
Triton oracle computes var/mean INSIDE the kernel, so should cuTile.

Exception: allocations and pure view/reshape/permute (metadata-only) are
fine on both sides. Explicit `torch.ops.prims.inductor_random` pre-gen is
also fine when Triton's eager fallback also uses it (mirror what Triton
does).

## Fixing procedure per oracle

1. Read the current cuTile port.
2. Read the Triton counterpart at `/home/dev/better-benchmark/repros/canonical/<name>/oracle.py`.
3. Compare kernel structure:
   - Number of `@ct.kernel` vs `@triton.jit`
   - Number of `ct.launch` vs Triton kernel-launch invocations
   - BLOCK sizes (from `@oracle_impl` kwargs)
   - What's inside each kernel body
4. Rewrite the cuTile port to mirror Triton's structure while removing
   antipatterns 1-2 above.
5. Validate: `python /home/dev/better-benchmark/scripts/validate_cutile_oracle.py <dir>`.
6. If numerics still pass, done. If they fail, revert.

## Time budget

~5 minutes per oracle. If a fix takes longer, move on.

## Report format

```json
{"batch": "fix_batch_XXX.txt", "fixed": N, "no_change_possible": N, "regressed": N,
 "results": {"<name>": "fixed:<antipattern>" | "no_change_possible:<reason>" | "regressed:<reverted>", ...}}
```
