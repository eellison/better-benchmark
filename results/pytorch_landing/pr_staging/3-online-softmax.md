# PR 3 — Fast combine for online-softmax loops

**Commit:** `a26fc2c8bf4` · **Flag:** `config.triton.online_softmax_fast_combine`.
**Status:** DEPENDS ON mega-commit online-softmax config + triton scaffolding — not standalone (conflicts on bare base in triton.py + config.py).

## Summary
Replaces the two-op online-softmax combine with a native `tl.max` on the
NaN-identical fast path.

## Measured impact
- **Kernel-geomean +0.83pp / per-model e2e +0.33pp.**
- genai: SoftmaxForward +10%, CrossEntropy +20%.

## Test plan
Softmax/CE genai micros; attention/softmax-bearing models. Flag-gated A/B.
Numerics-gated (NaN-identical path).

written with claude code
