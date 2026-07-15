# PR staging — prioritized Inductor perf commits

**Base:** `5e2ab3055de` · **Perf-branch HEAD:** `daa79cd25ca` · staged 2026-07-15.
All numbers from `results/pytorch_landing/LANDING_PRIORITY.md` (kernel-geomean
AND per-model-e2e; floors ±0.2% kernel / ±0.82pp model, n=158). **Nothing here
is pushed.** These are local artifacts for review before any upstreaming.

## Outcome: four merge-ready local branches, each rooted on the true base

Each branch below roots directly on `5e2ab3055de` and applies cleanly. rsqrt is
a single standalone commit; the other three needed a **minimal scaffolding
prerequisite commit** carved from the mega-commit `97385fb3273` before the PR
commit(s) cherry-pick clean on top. All branches live in `/tmp/pytorch-work`
(worktrees under `/tmp/pt-wt/`), local only.

| PR | branch | commits (base → head) | how made mergeable |
|----|--------|-----------------------|--------------------|
| **1. rsqrt canonicalization** | `pr/rsqrt-clean` | `a07f66f9996` | **CLEAN** — standalone cherry-pick of `a73d1583b34`, one file |
| **2. CE loop-invariant hoist** | `pr/ce-hoist-clean` | `1cca4ca8347` (prereq) → `8a52af079e5` | **carved-prereq-then-clean** — prereq carves the CE-online infra (prim + `CrossEntropyOnlineReduction` + Path-A lowering + postgrad fusion pass); `ab29345d82d` then cherry-picks clean and flips it to Path B. Companion `9b4edfffc15` **dropped** (see below) |
| **3. online-softmax fast combine** | `pr/online-softmax-clean` | `239a84e2550` (prereq) → `5205038a9ed` | **carved-prereq-then-clean** — prereq carves `scalar_reduction_accumulators` + the `reduction()` codegen rewrite (scalar/online-softmax path); `a26fc2c8bf4` then applies with one **textual-context** conflict resolved (kept `_online_softmax_fast_max`, dropped out-of-stack `has_store_with_rindex` context) |
| **4. scalar-acc + gate** | `pr/scalar-acc-clean` | `6b5a0810183` (prereq) → `5e81e8825b4` → `4a2db251d12` | **carved-prereq-then-clean** — prereq carves the same reduction() codegen rewrite + config flag + `MAX_R0_BLOCK` heuristics block; `ca8f961d6b0` + `9dde2c59a51` then cherry-pick clean |

### Conflict classification per PR (probed hunk-by-hunk)

- **rsqrt** — no conflict.
- **CE-hoist** — REAL SYMBOL DEPENDENCY: `cross_entropy_loss_online` is a lowering
  function introduced whole by the mega-commit; `ab29345d82d` is a *diff against
  it*. Carved the CE-online prereq so the pre-image exists; cherry-pick then
  zero-conflict. Note: `OnlineSoftmaxReduction` already exists at base — only the
  prim, `CrossEntropyOnlineReduction`, the `online_softmax_cross_entropy`
  reduction type, and the postgrad fusion pass are mega-introduced.
- **online-softmax** — mostly REAL SYMBOL DEPENDENCY (the scalar online-softmax
  `_block_max` codegen path a26fc patches is mega-introduced) plus one TEXTUAL
  CONTEXT drift in `TritonKernel.__init__`.
- **scalar-acc** — REAL SYMBOL DEPENDENCY: the whole `use_scalar_acc` codegen
  block + `scalar_reduction_accumulators` flag are mega-introduced; the two PR
  commits only append a gate condition / ungate + extra configs.

### Shared scaffolding (noted per the honesty gate)

PRs 3 and 4 both depend on the mega-commit's `reduction()` method rewrite in
`codegen/triton.py` (a single interleaved rewrite adding the scalar-accumulator
path, the scalar online-softmax path, and the `online_softmax_cross_entropy`
codegen). It is **not git-separable** into finer pieces — carried whole in each
prereq. All helper symbols it calls already exist at base. Each branch carries
its own independent copy of the prereq it needs (3 and 4 overlap on the triton
rewrite; 4 additionally needs the heuristics `MAX_R0_BLOCK` block). If these are
upstreamed together, the two prereqs should be merged into one shared
scaffolding PR that both stack on.

### Companion `9b4edfffc15` (CE target_logit hoist) — DROPPED

It patches the in-loop `online_softmax_cross_entropy` triton codegen, but
`ab29345d82d`'s "Path B" (and HEAD `daa79cd25ca`) never creates that reduction —
the lowering uses `OnlineSoftmaxReduction` + a separate `Pointwise` gather — so
the codegen the companion touches is dead under Path B. It also bundles an
unrelated `slice_scatter_partial_elision` change (config.py + a
`slice_scatter_elision.py` that does not exist at base). Not separable, not
needed for the CE-hoist win; excluded.

## Verification (per branch)

For every branch: `git log --oneline 5e2ab3055de..<branch>` shows exactly the
intended commits; `git grep -c '^<<<<<<<'` is **0**; **every touched `.py`
ast-parses** (`git show <branch>:<file> | python -c 'import ast,sys;
ast.parse(sys.stdin.read())'`); and each symbol the PR references is confirmed
defined in the branch tree by grep. Each regenerated patch was `git apply
--check`'d onto a pristine `5e2ab3055de` checkout and applies clean.

**Caveat (stated honestly):** this environment is CPU-only with no PyTorch
build, so nothing was compiled or run. The bar met is: clean apply on base +
ast-parse of every touched file + symbol-presence grep + internal consistency
(e.g. carried config flags referenced by carried code are defined). Runtime /
numerics are NOT verified here.

## Artifacts

- `patches/` — `git format-patch` of each branch (base-rooted; regenerated
  2026-07-15 from the verified branches, each verified to `git apply --check`
  clean onto `5e2ab3055de`):
  - `1-rsqrt.patch` — 1 commit
  - `2-ce-hoist.patch` — 2 commits (prereq + PR)
  - `3-online-softmax.patch` — 2 commits (prereq + PR)
  - `4-scalar-acc.patch` — 3 commits (prereq + gate + ungate)
- **Verified branches** in `/tmp/pytorch-work`: `pr/rsqrt-clean`,
  `pr/ce-hoist-clean`, `pr/online-softmax-clean`, `pr/scalar-acc-clean`.
- Per-PR descriptions: `1-rsqrt.md` … `4-scalar-acc.md` (below).

## Recommended upstreaming order

1. **rsqrt** first — clean, standalone, ~half the branch's model-level win
   (+2.11pp e2e), 45-line peephole. No dependencies.
2. Then a **shared scaffolding PR** = the merged prereq (CE-online infra +
   scalar/online-softmax `reduction()` codegen rewrite + config flags +
   heuristics block) that the other three stack on.
3. Then CE-hoist, online-softmax-fast-combine, scalar-acc (gate+ungate) on top.
