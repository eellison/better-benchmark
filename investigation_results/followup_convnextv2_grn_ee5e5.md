# Follow-up (FILE ONLY): ConvNeXtV2 GRN reduction (sum_sum_sum_ee5e53038768)

Status: LOGGED 2026-06-16. Migration row is DONE (oracle_measured, AT_FLOOR
128.67us vs compile 130.75us, all checks pass). This note records two findings
from the long needs_work history — neither blocks the row.

## Background
This ConvNeXtV2 GRN/GELU repro returns: f32 channel reductions (out 0,1),
scalar zero (out 2), a DENSE bf16 tensor [128,2560,7,7] (out 3), and a final
f32 channel reduction OVER that rounded dense tensor (out 4). It was bounced to
needs_work many times: fast pure-Triton variants failed dense output 3
(max_diff 8/64/128 — wrong fast path), while faithful variants ran slower than
compile (~193us vs ~163us). The committed candidate finally mirrors the
compiled multi-output schedule faithfully AND lands AT_FLOOR.

## Finding 1 — gap class is SCHEDULER_FUSION, near-floor (not a compile bug)
Inductor already emits a near-floor multi-kernel schedule for this shape; the
oracle can only MIRROR it, not beat it (oracle 128.67 vs compile 130.75us,
ratio 1.016). The difficulty was faithfulness, not speed headroom. The
docstring fix direction: represent this recurrent GRN/GELU multi-output
schedule as a reusable guarded fusion.

## Finding 2 — compile is LESS numerically accurate than eager/oracle here
fp64-anchored (this B200):
- output 3 (dense bf16): oracle/eager vs fp64 = 16 ; **compiled vs fp64 = 2.62e5**
- output 4 (sum over dense): oracle vs fp64 = 0.5 ; **compiled vs fp64 = 4.17e5**
i.e. torch.compile's schedule for the dense bf16 store + downstream sum is
markedly less accurate than eager (and than the oracle) at these magnitudes —
likely a bf16 accumulation/rounding-order choice in the generated kernel. The
oracle is correct; this is a compile-side numerics observation worth a look
(does Inductor pick a worse accumulation order / rounding for the GRN dense
store?). Not actioned.

Cross-ref: this is the same bf16-large-magnitude reduction-accuracy theme as
nvp_triage_2026-06-16.md (the --check bf16 tolerance issue).
