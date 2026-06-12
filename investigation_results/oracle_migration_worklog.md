# Oracle Migration Worklog

Updated: 2026-06-12

## Current Worker Pool

- Row 122 `pointwise_88bffcefddc4`: active worker `019ebd71-bd99-7ed3-9664-561ca57b630e`.
- Row 123 `sum_sum_cd8694c00507`: active worker `019ebd73-a5cc-7672-95d6-b8cd6a888e02`.
- Row 124 `var_mean_88858c55c3b4`: active worker `019ebd74-7360-7202-89cd-ec88aae96af5`.
- Row 125 `pointwise_182f6f9450b9`: active worker `019ebd75-6886-78e3-bf4e-af115c359931`.

Refill buffer owned by this session: rows 141-145. Rows 126-140 were claimed remotely by other `Codex` batches and should not be reused by this session unless ownership changes.

## Pending Parent Review

- Row 94 `var_mean_5a22dd21d88e`: worker checks passed; fallback bench was `23GOOD_1BAD_ORACLE`.
- Row 112 `var_mean_45f7dfd4a983`: worker checks passed; fallback bench was `4GOOD_1AT_FLOOR`.
- Row 113 `pointwise_35ecf6633bb0`: worker checks passed; parent fallback bench was `1GOOD_1AT_FLOOR`.
- Row 115 `sum_e529e567d636`: worker checks passed; fallback bench was `1GOOD_1AT_FLOOR`.
- Row 121 `sum_sum_sum_e7781939b0a2`: worker checks passed; fallback bench was `1GOOD_1AT_FLOOR`.

## Pushed By This Session

- Row 61 `var_mean_2e254a2827d8`: measured, 22/22 checks, H100 fallback bench `20GOOD_1AT_FLOOR_1BAD_ORACLE`.
- Row 70 `sum_785c25a716ed`: measured after rework, 3/3 checks, H100 fallback bench `3GOOD`; duplicate shape registration warning is expected for same-shape/different-stride ConvBERT points.
- Row 81 `sum_4a4493837e6e`: measured, 16/16 checks, H100 fallback bench `15GOOD_1AT_FLOOR`.
- Row 84 `pointwise_1c9e8dc48812`: marked `needs_work`; eager-matching oracle fails FP64 bench gate because compiled uses fp32 RoPE arithmetic.
- Row 85 `pointwise_a77badc5e988`: marked `needs_work`; checks pass but FP64 bench gate is NaN/invalid with no timing.
- Row 86 `pointwise_bd0149b22f68`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 87 `pointwise_e52ac85e10fc`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 88 `var_mean_eac408f45b9d`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 89 `sum_7cfdb80c54c5`: measured, 23/23 checks, H100 fallback bench `21GOOD_2AT_FLOOR`.
- Row 90 `sum_6d68a671ec4a`: measured, 5/5 checks, H100 fallback bench `4GOOD_1AT_FLOOR`.
- Row 91 `sum_sum_sum_c5cdd9ab78b4`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 92 `var_mean_ec0f56a425b2`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 93 `pointwise_e6ddc8e897ec`: marked `needs_work`; 24/24 checks pass but locked bench fails the FP64 numerics gate with NaN errors and emits no timings.
- Row 95 `pointwise_c509446d4a84`: measured, 3/3 checks, H100 fallback bench `3GOOD`.
- Row 111 `sum_sum_4c6ae5dbcf21`: measured, 2/2 checks, H100 fallback bench `2GOOD`.
- Row 114 `pointwise_733dafce05a6`: measured, 2/2 checks, H100 fallback bench `2GOOD`.

All H100 fallback rows still need native B200 measurement before treating timings as official B200 floors.

## Notes

- Parent verification is required before marking a worker row measured.
- Remote manager commits may convert H100 fallback successes to B200 `needs_work`; preserve those remote statuses during rebase.
- Do not commit active worker oracle files until their workers report completion and parent verification passes.
