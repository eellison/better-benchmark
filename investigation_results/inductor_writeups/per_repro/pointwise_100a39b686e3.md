# pointwise_100a39b686e3


## Measured Timings
- Oracle: 3.81 us
- Compile (CDT): 5.02 us
- Ratio: 1.32x

## Current Result

- Family: `layout_indexing_stencil_fusion` from the expanded worklist, but the
  actual repro is a scalar in-place pointwise update.
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_100a39b686e3/oracle_layout_stencil.py`
- Correctness: PASS
- True floor: no, `not_true_floor`

## Diagnosis

The full compiled scope is:

```python
add_tensor = torch.ops.aten.add.Tensor(arg560_1, 1)
copy__default = torch.ops.aten.copy_.default(arg560_1, add_tensor)
return copy__default
```

`arg560_1` is a CUDA `int64[]` scalar. The oracle keeps the same scope by
launching one Triton pointwise kernel that loads the scalar, adds one, stores it
back to the same input tensor, and returns that mutated tensor. The checker
clones inputs for eager and oracle separately so the in-place `copy_` semantics
are actually validated: output value, mutated input value, and returned-output
aliasing all pass.

This is not a layout/stencil fusion opportunity despite the inferred family.
The workload moves 16 bytes and already runs as one launch. The large SOL gap is
an artifact of `sol_us=0.0`/tiny-byte accounting, not useful remaining GPU work.
No Inductor scheduler change is indicated for this repro.

## Measurements

Commands:

```bash
python -m py_compile repros/canonical/pointwise_100a39b686e3/oracle_layout_stencil.py
python repros/canonical/pointwise_100a39b686e3/oracle_layout_stencil.py --check
python repros/canonical/pointwise_100a39b686e3/oracle_layout_stencil.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/pointwise_100a39b686e3/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_required --n-warmup 10 --n-rep 50 --rounds 5 --max-workers 1 --output /tmp/pointwise_100a39b686e3_bench_compare.json
python scripts/validate_corpus_invariants.py
```

Results:

- `--check`: PASS.
- Check details: output 0 exact PASS, input mutation PASS, output alias PASS.
- `--bench --warmup 10 --rep 50`:

```json
{"repro_id": "pointwise_100a39b686e3", "oracle_us": 3.808, "compile_us": 3.456, "combo_compile_us": 3.488, "best_required_compile_us": 3.456, "historical_best_compile_us": 5.024000070989132, "ratio": 0.908, "status": "AT_FLOOR", "classification": "BANDWIDTH_BOUND", "true_floor": false}
```

- Required compile-only interleaved comparison from `bench_compare.py`:
  `cd=3.9679999463260174 us`, `combo_required=4.000000189989805 us`,
  `total_bytes=16`, `rep_per_round=500`.
- Historical best from `interleaved_3config_results.csv`:
  `5.024000070989132 us`; worklist rounded value is `5.0 us`.
- Corpus invariants: PASS, all hard invariants satisfied.

## CSV Notes

`full-scope --check PASS; measured_oracle_us=3.808; cd_compile_us=3.456; combo_compile_us=3.488; best_required_compile_us=3.456; interleaved_cd_compile_us=3.9679999463260174; interleaved_combo_compile_us=4.000000189989805; historical_best_compile_us=5.024000070989132; classification=BANDWIDTH_BOUND; true_floor=no; not_true_floor; scalar int64 in-place add/copy_ is already launch-floor; owner=Codex-template-layout-100a`
