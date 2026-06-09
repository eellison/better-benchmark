# Oracle Dispatch: per-shape, per-hardware registration

## Problem

All 1483 oracles were written and tuned on **H100**, each at **one captured
shape**. We measure on B200 (and eventually other GPUs / other shapes). In the
fresh B200 measurement (`results/all_oracle_timings_b200_v2.json`), 474 of
1409 oracles came out BAD_ORACLE (slower than compile) — many simply because
H100-tuned BLOCK sizes / num_warps don't transfer to B200, not because the
algorithm is wrong. We need a way to:

1. record what each oracle was actually written for (hardware + shape),
2. add variants (new configs or new kernels) for other hardware/shapes
   without forking oracle files,
3. have benchmarks honestly report when they're running a kernel outside its
   tuned operating point.

## Design

### One concise explicit line per oracle

```python
from oracle_harness import oracle_impl

@oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)")
def oracle_forward(inputs): ...
```

Design decisions (each iterated with the user):

- **Explicit, not implicit.** Zero-line "the harness infers it" magic was
  rejected: declaring what an oracle was written for is valuable
  documentation in the file itself. But it must be ONE line — per-file
  registry instantiation ceremony is an anti-pattern.
- **`shapes` is the FULL input signature**, not a single tensor shape. An
  oracle implements the whole repro (multiple tensors, possibly multiple
  kernels); its operating point is the complete signature. The format is the
  existing `T()/S()` string — copy `_shapes_config` from the repro verbatim.
  `S()` entries (shape params) are skipped; only tensors matter.
- **dtype is documentation, not a match key.** The corpus dedupes patterns
  across dtypes — the same oracle gets f32 inputs when written against bf16,
  so requiring dtype match would leave most inputs unmatched. Matching is
  shape-only, BUT dispatch flags `dtypes_differ` (with tuned vs actual
  dtypes) in its info and in the bench JSON, so if dtype ever matters for
  floors, the data will show it.
- **Matching is exact-only** (footgun protection, see below).

### Shared kernels, different configs

The same kernel body is often right for many (hardware, shape) points with
different launch configs. `configs` is passed through as kwargs at dispatch;
the decorator returns the function unchanged so it can be registered N times:

```python
def _softmax_impl(inputs, *, BLOCK, num_warps): ...

SIG = "(T([32768, 1024], bf16),)"
oracle_impl(hardware="H100", shapes=SIG, BLOCK=1024, num_warps=4)(_softmax_impl)
oracle_impl(hardware="B200", shapes=SIG, BLOCK=2048, num_warps=8)(_softmax_impl)
```

There is no blessed `configs` dict — any keyword argument beyond
`hardware`/`shapes`/`description` is passed straight to the implementation
at dispatch. That includes algorithm-strategy flags, not just numeric knobs:

```python
oracle_impl(hardware="H100", shapes=SIG, persistent=True,  RBLOCK=512)(_reduction)
oracle_impl(hardware="B200", shapes=SIG, persistent=False, RBLOCK=128)(_reduction)
```

A genuinely different algorithm is just another decorated function.

### Exact-only matching — no fuzzy shape fallback

There is no continuous shape space to interpolate over: each repro has a
small, known, finite set of shapes (its shapes.txt lines, typically 5-15).
Fuzzy "nearest shape" matching would solve a problem we don't have while
creating a real footgun — silently running a kernel with a hardcoded `BLOCK`
dividing rnumel (or a persistent kernel that only fits small rnumel) at a
different shape either crashes or, worse, produces a garbage "floor" that
poisons gap data.

So an implementation either:
- **exactly matches** the runtime signature (it was registered for it), or
- declared **`shapes=None`** — shape-general: grid computed from input dims,
  works anywhere.

A kernel tuned at 3 of the 12 shapes.txt lines gets 3 registrations (same
body, different `configs`). **No match → `OracleDispatchError`**;
`bench_oracle` reports `"status": "NO_ORACLE_FOR_SHAPE"` instead of a fake
ratio. A loud failure beats a quiet wrong number.

### Match order

| `matched` | Meaning |
|-----------|---------|
| `"hardware+shape"` | signature exact AND tuned on this GPU — trustworthy floor |
| `"shape"` | signature exact, tuned on other hardware (e.g. H100 kernel on B200) |
| `"hardware"` | shape-general impl for this GPU |
| `"any"` | shape-general, unconstrained |
| — | nothing matches → `OracleDispatchError` |

First match wins; registration order breaks ties. `fallback` is true for
anything but `"hardware+shape"`.

Today, every migrated H100 oracle measured on B200 reports
`"matched": "shape"` — the honest description of the entire current corpus.

### bench_oracle integration

`bench_oracle()` resolves dispatch automatically when the oracle's module has
`oracle_impl` registrations, and adds to the result JSON line:

```json
"dispatch": {"matched": "shape", "tuned_on": "H100", "running_on": "B200",
             "fallback": true}
```

Unmigrated modules (no registrations) behave byte-identically to before — no
dispatch field. `bench_oracle_all_shapes` re-resolves per shape entry, so a
file with per-shape variants automatically uses the right one per line of
shapes.txt.

## Migration

Mechanical per oracle file: add the import and one decorator line with the
repro's `_shapes_config` first entry verbatim:

```python
@oracle_impl(hardware="H100", shapes="<_shapes_config string>")
def oracle_forward(inputs): ...
```

Pilots verified on B200 (timing unchanged, dispatch field present):
- `amax_sum_3ed297ef02cd/oracle_online_softmax.py`
- `sum_e00c7291b6ee/oracle_cooperative_split_k.py`

Then, to attack the 474 BAD_ORACLEs: add B200 `configs` variants to the same
kernel bodies and re-measure — the registration mechanism makes each fix a
2-line diff.

## Self-test

`python scripts/test_oracle_dispatch.py` — covers signature parsing
(including S()-skipping and no-trailing-comma single-tensor strings), every
match level, wrong-shape raising, exact-beats-general precedence, per-shape
configs on one body, multi-tensor signatures, configs passthrough,
dtype-agnostic matching, and unmigrated-module passthrough. No GPU required.
