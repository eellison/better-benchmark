# Opened draft PRs on pytorch/pytorch (2026-07-16, agent-submitted per user authorization)

All four are DRAFT, [WIP]-titled, with the agent-submitted disclosure block at the top of the body.

| # | source branch | fork branch | PR | rebase outcome |
|---|---|---|---|---|
| 1 | pr/rsqrt-clean | eellison:bb/rsqrt-canonicalization | https://github.com/pytorch/pytorch/pull/190206 | rebased clean onto origin/main (e55be5fb1ae) |
| 2 | pr/scalar-acc-clean | eellison:bb/scalar-acc-reductions | https://github.com/pytorch/pytorch/pull/190207 | rebase conflicted (triton_heuristics.py `_reduction_configs` refactored on main) → aborted, pushed base-rooted (5e2ab3055de); noted in body |
| 3 | pr/online-softmax-clean | eellison:bb/online-softmax-fast-combine | https://github.com/pytorch/pytorch/pull/190208 | rebased clean onto origin/main (e55be5fb1ae) |
| 4 | pr/mor-finalize-clean | eellison:bb/mor-finalize-sum | https://github.com/pytorch/pytorch/pull/190209 | rebased onto origin/main (e55be5fb1ae); one trivial context conflict in simd.py (`nsplit` sizevar assertion refactored on main) resolved — kept main's AssertionError form + the branch's `nsplit_expr` |

Notes:
- Bodies are minimal per user directive (disclosure block, one-paragraph summary, impact line, load-bearing caveat, attribution).
- Nothing pushed to pytorch/pytorch directly; fork branches only, PRs opened via gh as eellison.
- /tmp/pytorch-work HEAD (work2 @ daa79cd25ca) untouched; throwaway worktrees under /tmp/bb-pr-worktrees removed after use; no force-pushes.

## Update 2026-07-16: #190208 rebased onto main (post-fmax #189162)
fmax merged to main (2eaa680c065) — overlaps only the two-pass fallback, NOT our online-combine loop. Rebased bb/online-softmax-fast-combine onto main+fmax (real rebuild, torch 2.14), dropped the redundant two-pass hunk. **Win survives: 1.24x geomean on the 5 online-softmax repros vs main+fmax** (amax_sum_02064a1e60ac 1.99x). PR #190208 force-pushed + body updated. Still draft/WIP.
