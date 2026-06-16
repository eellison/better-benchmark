# Finding E: does a dynamic capture dedup to the static capture of the same hint shapes?

Date: 2026-06-16 (adversarial review round 1, deferred deep finding).
Status doc §E claimed: "a dynamic capture dedupes against the static capture
of the same hint shapes — shape_hash is computed from hint ints, the
symbolic block is metadata." **This claim is currently FALSE.** This doc has
the measured evidence and the design decision needed.

## Evidence (measured, GroupNorm/var_mean, hint 64×64×16×16)

Capturing the SAME module dynamic vs static:

| | pattern_hash | shape_hash | inputs |
|---|---|---|---|
| dynamic | `3a40eaee0716` | `2e02cdda` | `[tensor[64,64,s53,s0], I(256,"s0*s53"), I(16,"s53"), I(16,"s0"), w, b]` |
| static  | `663c5e3067ba` | `7395c806` | `[tensor[64,64,16,16], w, b, S[64,32,2,256], S[64,64,16,16]]` |

BOTH hashes fork. Two distinct causes:

1. **shape_hash forks because of the symint INPUT placeholders.** The
   inline-reshape change (§2.5b, needed for ShapesSpec reuse) lifts the
   symbolic dims as symint inputs (`mul`, `s0`, `s53`); `_record_placeholder`
   puts them in `placeholder_info` with `dtype:"symint"`;
   `shape_hash_for_placeholders` hashes every placeholder, so the 3 extra
   entries inflate the dynamic hash. The static capture has zero symint
   placeholders (it lifts shapes as `_shape_param` list args, which are
   `shape_params`, NOT in `placeholder_info`).
   MEASURED: excluding `dtype=="symint"` placeholders from the hash makes the
   dynamic hash become `7395c806` == the static hash EXACTLY.
   SAFE: scanned all 1727 corpus shapes.json — ZERO have a sym/I input entry,
   so excluding symints changes no existing static hash.

2. **pattern_hash forks because the GRAPHS are genuinely different.** Dynamic
   = inline `reshape(x, [64,32,2,mul])` + 3 symint inputs; static = lifted
   `reshape(x, _shape_param_0)` + 2 list params. Plus dynamic subgraphs SKIP
   canonicalization (`[canonicalize] symint placeholder ... retrace skipped`,
   capture_hook ~:1029), so even the structural normalization differs.

3. **Corollary — non-deterministic identity.** Symbol allocation is
   trace-context-sensitive: re-capturing the same module scattered into
   shape `2e02cdda`/{s0,s53} vs `04a2312b`/{s13,s21}, different pattern_hash,
   even a different partition. Two runs of the same dynamic model land in
   different canonical dirs.

## The real question: SHOULD they dedup?

A dynamic capture and a static capture of the same kernel are, by the
feature's own premise (§1.4), DIFFERENT kernels — static = 1 persistent
reduction @13.7us, dynamic = 2 looped @35.5us, the 2.6x gap. So arguably
they SHOULD be distinct corpus entries — the §E "dedup" claim may be the
wrong goal. But the accounting use case wants to JOIN them ("for this
kernel, static floor vs dynamic floor"), which needs a shared key.

Two coherent designs:

### Option 1 — they are DISTINCT points under ONE pattern, joined by a kernel key
Keep dynamic and static as separate shape points, but make them share the
**pattern** so a join is possible. Requires: (a) exclude symint placeholders
from shape_hash (safe, measured); (b) canonicalize dynamic subgraphs so the
inline-reshape form normalizes to the same pattern_hash as the static
lifted form (currently skipped — the hard part); (c) accept that shape_hash
still differs (symint inputs vs not is a real input-signature difference) OR
also fold the inline-reshape inputs to match. A `captured_dynamic` flag on
the point already distinguishes them. This makes "same pattern, static point
+ dynamic point" the join unit.

### Option 2 — dynamic is a SEPARATE family entry; join by (origin_ops, models)
Accept that dynamic captures get their own pattern/shape hashes (they ARE a
different graph), and do the static-vs-dynamic join at the ACCOUNTING layer
via the shared `origin_ops` multiset + `models` keys (which both captures
carry), not via hash equality. No hashing change; the §E claim is retracted
and replaced with "dynamic and static are distinct entries, correlated by
model+op-set in accounting." Lowest-risk; defers the join to where it's
actually consumed.

## Recommendation
Option 2 for now (retract the §E dedup claim; correlate at accounting),
PLUS the safe, isolated shape_hash change (exclude symint placeholders) so
that IF two dynamic captures of the same family occur, their shape_hash
reflects only the real tensor signature, not the incidental symint-input
count. Option 1's dynamic-subgraph canonicalization is a larger project
(the retrace-skip exists because make_fx on a symint-input graph is not yet
handled) — separate work, not blocking.

NEEDS USER DECISION: Option 1 (make them dedup, canonicalize dynamic
subgraphs) vs Option 2 (distinct entries, join in accounting). The status
doc §E claim must be corrected either way.
