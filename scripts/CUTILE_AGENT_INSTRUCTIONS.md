# cuTile port subagent instructions

You are converting a batch of Triton oracles to cuTile oracles.

## Your input

* A batch file at `scripts/cutile_batches/batch_XXX.txt` containing one
  canonical repro name per line (~50 total).

## Your job for each name

1. Read the Triton oracle at `repros/canonical/<name>/oracle.py`.
2. Read the repro at `repros/canonical/<name>/repro.py` — this defines the
   `Repro` module the oracle must match numerically.
3. Read `repros/canonical/<name>/shapes.json` for input shape metadata.
4. Write a cuTile port at `repros_cutile/canonical/<name>/oracle.py`.
   - The mirror dir already contains symlinks to `repro.py`, `shapes.json`,
     `meta.json`. You ONLY write `oracle.py`.
5. Validate the port:
   ```
   python scripts/validate_cutile_oracle.py repros_cutile/canonical/<name>
   ```
   If numerics fail or import errors, either fix the port or (last resort)
   convert it to a `NotImplementedError` stub with a one-line reason.

## Rules

* Read `scripts/cutile_reference.md` FIRST — it lists Triton→cuTile
  equivalences, gotchas, and a working template.
* Copy the `@oracle_impl(...)` decorators verbatim from the Triton oracle,
  BUT drop `num_warps=`, `num_stages=` kwargs — cuTile doesn't accept them,
  and they'll be forwarded to your `oracle_forward` (breaking it). Keep only
  kwargs your function actually uses (BLOCK, BLOCK_M, etc.).
* Preserve the docstring's classification tag (SCHEDULER_FUSION, etc.) if
  present, but replace the description with a one-line summary of the cuTile
  port.
* Prefer the SIMPLEST cuTile implementation that passes the numerics gate.
  Do not attempt to autotune, add cooperative reductions, or use advanced
  APIs unless the Triton kernel does.
* If cuTile lacks a feature the Triton kernel needs (seeded RNG,
  inline PTX asm, custom rounding modes, masked stores) and you can't work
  around it, write a stub that raises `NotImplementedError("cuTile port
  unsupported: <reason>")`. Do NOT copy Triton code as-is.
* Do NOT modify anything outside `repros_cutile/canonical/`.

## Working examples

Read these five hand-written cuTile ports for reference patterns:
* `repros_cutile/canonical/sum_9552a61d796d/oracle.py` — bf16 reduction
* `repros_cutile/canonical/mean_db9733790220/oracle.py` — row RMSNorm
* `repros_cutile/canonical/pointwise_000209e1748d/oracle.py` — layout clone
* `repros_cutile/canonical/pointwise_0a49dc00cbaf/oracle.py` — scalar fill
* `repros_cutile/canonical/pointwise_100a39b686e3/oracle.py` — in-place +1

## Validation pattern

After writing an oracle:

```bash
python scripts/validate_cutile_oracle.py repros_cutile/canonical/<name>
```

Expect a single JSON line. If `"numerics_ok": true`, move on.
If `"numerics_ok": false` or `"load": "error"`, look at the error and fix.
If truly unportable, replace with a `NotImplementedError` stub.

## Progress reporting

Keep a running tally. At the end, write a JSON summary line (no other
output) with `name -> "ok" | "not_implemented" | "failed:<reason>"`.

Example final line:
```
{"batch": "batch_003.txt", "ok": 42, "not_implemented": 6, "failed": 2,
 "results": {"pointwise_abc123": "ok", "sum_def456": "not_implemented:tl.rand",
             "amax_ghi789": "failed:reshape mismatch"}}
```

## Efficiency tips

* Many oracles are similar (small variations of RMSNorm, elementwise, etc.).
  Once you have a pattern working, reuse it verbatim for cousins.
* When multiple `@oracle_impl(...)` decorators stack on ONE function,
  keep them stacked in the cuTile port too (drop only the incompatible
  kwargs).
* The pilot cuTile oracles above cover the common patterns — start by
  reading them.
* Don't over-engineer BLOCK sizes; use the Triton BLOCK values as a
  starting point and trust the numerics gate.

## Time budget

Aim to complete your batch in reasonable wall-clock time. If any single
port takes more than ~5 minutes of iteration, stub it with
`NotImplementedError` and move on.
