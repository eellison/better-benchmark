# PR staging — prioritized Inductor perf commits

**Base:** `5e2ab3055de` · **Perf-branch HEAD:** `daa79cd25ca` · staged 2026-07-15.
All numbers from `results/pytorch_landing/LANDING_PRIORITY.md` (kernel-geomean
AND per-model-e2e; floors ±0.2% kernel / ±0.82pp model, n=158). **Nothing here
is pushed.** These are local artifacts for review before any upstreaming.

## The key finding: only rsqrt is independently landable

Probed each commit's cherry-pick onto the bare base. Result:

| PR | commit(s) | picks clean on base? | dependency |
|----|-----------|----------------------|------------|
| **1. rsqrt canonicalization** | `a73d1583b34` | ✅ **CLEAN** | none — standalone, one file |
| 2. CE loop-invariant hoist | `ab29345d82d` (+`9b4edfffc15`) | ❌ conflict | needs `inductor_prims.cross_entropy_loss_online` + `online_softmax_cross_entropy` + `OnlineSoftmaxReduction` — all introduced by the **mega-commit `97385fb3273`** |
| 3. online-softmax fast combine | `a26fc2c8bf4` | ❌ conflict | needs the online-softmax config + triton scaffolding from the mega-commit |
| 4. scalar-acc + gate | `ca8f961d6b0` + `9dde2c59a51` | ❌ conflict | needs `scalar_reduction_accumulators` (config + triton_heuristics), introduced by the mega-commit |

**Upstreaming consequence:** rsqrt lands as-is against mainline PyTorch today.
The other three are *real, measured wins* but sit on top of the mega-commit's
CE-online / scalar-accumulator infrastructure — they cannot be cherry-picked as
standalone PRs. To upstream them you must either (a) land a minimal slice of the
mega-commit's prim/config scaffolding first (a prerequisite PR), then stack
these on top, or (b) squash each win together with the scaffolding it needs into
one self-contained PR. This is a genuine sequencing constraint the per-commit
attribution surfaced — not a staging failure.

## Artifacts

- `patches/` — `git format-patch` of every unit (base-independent record of the
  exact diff, applies onto HEAD-lineage; `git am` or `git apply`):
  - `1-rsqrt.patch` — the standalone win
  - `2-ce-hoist.patch`, `2b-ce-target-logit-hoist.patch`
  - `3-online-softmax-fastcombine.patch`
  - `4a-scalar-acc-gate.patch`, `4b-scalar-acc-ungate.patch`
- **Verified branch:** `pr/rsqrt-clean` in /tmp/pytorch-work — 5e2ab + rsqrt,
  cherry-pick clean, `post_grad.py` parses. The only branch that is a
  ready-to-push PR as-is.
- Per-PR descriptions: `1-rsqrt.md` … `4-scalar-acc.md` (below).

## Recommended upstreaming order

1. **rsqrt** first — clean, standalone, ~half the branch's model-level win
   (+2.11pp e2e), 45-line peephole. No dependencies.
2. Then a **CE-online / scalar-acc scaffolding PR** (minimal slice of the
   mega-commit) as the base the rest stack on.
3. Then CE-hoist, online-softmax-fast-combine, scalar-acc-gate on top.
