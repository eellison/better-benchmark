# amax_sum_9940b361e5b4

## Summary

- Family: `online_softmax_cross_entropy`
- Claim/owner: `Codex-template-online-9940`
- Oracle: `repros/canonical/amax_sum_9940b361e5b4/oracle_online_softmax.py`
- Classification: `NEW_PATTERN`
- True floor: yes

## Gap Diagnosis

The repro is a full Longformer inference sliding-window attention path, not a
standalone softmax. It diagonalizes `bmm_22`, builds the key-mask bias from
`arg7_1`, applies the query mask from `arg8_1`, softmaxes each 513-wide local
attention row, then writes the padded/shifted `[192, 256, 768]` output layout
with stride `(197120, 769, 1)`.

The oracle computes the same full `Repro()(*make_inputs())` scope directly in
Triton: one kernel zeroes the padded destination storage and one row kernel
reconstructs the Longformer band scores, applies fp16-equivalent mask behavior,
uses a scalar-accumulator row softmax, and stores directly into the final
strided layout.

Inductor cannot do this today because the scheduler sees many generic
view/pad/slice/slice_scatter/select_scatter operations around a generic
`amax -> exp -> sum -> div` softmax. It does not have a Longformer sliding-window
attention pattern that fuses the structured band assembly with the softmax
reduction epilogue and final layout scatter.

## Validation

Correctness:

```bash
python repros/canonical/amax_sum_9940b361e5b4/oracle_online_softmax.py --check
```

Result:

```text
Checking amax_sum_9940b361e5b4...
  output 0: PASS (shape=[192, 256, 768] dtype=torch.float16 stride=(197120, 769, 1) max_diff=0.00e+00 max_rel=0.00e+00)
Correctness: PASS
```

Oracle timing:

```bash
python repros/canonical/amax_sum_9940b361e5b4/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Result:

```json
{"classification": "NEW_PATTERN", "compile_us": null, "historical_best_compile_us": 340.86400270462036, "oracle_min_us": 235.072, "oracle_us": 237.376, "ratio": null, "rep": 50, "repro_id": "amax_sum_9940b361e5b4", "scope": "full_repro_forward", "status": "GOOD", "true_floor_vs_historical_best": true, "warmup": 10}
```

Local compile comparison:

```bash
python scripts/bench_compare.py repros/canonical/amax_sum_9940b361e5b4/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --max-workers 1 --output /tmp/amax_sum_9940b361e5b4_compare.json
```

Result:

```text
cd=391.4240002632141 us
combo=285.0239872932434 us
historical_best_compile_us=340.86400270462036 us
oracle_us=237.376 us
```

CSV notes: `oracle_us=237.376`, `oracle_min_us=235.072`,
`cd_compile_us=391.4240002632141`, `combo_compile_us=285.0239872932434`,
`historical_best_compile_us=340.86400270462036`, `classification=NEW_PATTERN`,
`true_floor=yes`, `scope=full_repro_forward`.
