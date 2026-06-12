# Oracle Migration Worklog

Updated: 2026-06-12

## Current Worker Pool

- Row 89 `sum_7cfdb80c54c5`: active worker `019ebd48-7905-7141-b806-cbcf1243e864`.
- Row 70 `sum_785c25a716ed`: active rework worker `019ebd4e-6452-7f12-b383-60f0b1720dfd`; previous parent bench was correct but slower than compiled on H100 fallback.
- Row 93 `pointwise_e6ddc8e897ec`: active worker `019ebd51-0563-7001-a264-2a2145aa0c6b`.
- Row 94 `var_mean_5a22dd21d88e`: active worker `019ebd53-2fc1-7032-a532-fdc9babadeb3`.
- Row 111 `sum_sum_4c6ae5dbcf21`: active worker `019ebd5f-a9c4-7012-a06d-4feecc96b97d`.

Refill buffer owned by this session: rows 112-115. Rows 116-120 were claimed remotely by another `Codex` batch and should not be reused by this session unless ownership changes.

## Pushed By This Session

- Row 61 `var_mean_2e254a2827d8`: measured, 22/22 checks, H100 fallback bench `20GOOD_1AT_FLOOR_1BAD_ORACLE`.
- Row 81 `sum_4a4493837e6e`: measured, 16/16 checks, H100 fallback bench `15GOOD_1AT_FLOOR`.
- Row 84 `pointwise_1c9e8dc48812`: marked `needs_work`; eager-matching oracle fails FP64 bench gate because compiled uses fp32 RoPE arithmetic.
- Row 85 `pointwise_a77badc5e988`: marked `needs_work`; checks pass but FP64 bench gate is NaN/invalid with no timing.
- Row 86 `pointwise_bd0149b22f68`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 87 `pointwise_e52ac85e10fc`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 88 `var_mean_eac408f45b9d`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 90 `sum_6d68a671ec4a`: measured, 5/5 checks, H100 fallback bench `4GOOD_1AT_FLOOR`.
- Row 91 `sum_sum_sum_c5cdd9ab78b4`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 92 `var_mean_ec0f56a425b2`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 95 `pointwise_c509446d4a84`: measured, 3/3 checks, H100 fallback bench `3GOOD`.

All H100 fallback rows still need native B200 measurement before treating timings as official B200 floors.

## Notes

- Parent verification is required before marking a worker row measured.
- Remote manager commits may convert H100 fallback successes to B200 `needs_work`; preserve those remote statuses during rebase.
- Do not commit active worker oracle files until their workers report completion and parent verification passes.
