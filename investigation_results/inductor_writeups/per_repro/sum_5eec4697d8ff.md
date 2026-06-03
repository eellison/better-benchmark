# sum_5eec4697d8ff

## Current Result

- Family: `structured_pool_upsample_backward_reduce`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_5eec4697d8ff/oracle_longformer_scatter_reduce.py`
- Correctness: PASS
- Oracle: `65.632 us`
- `torch.compile coordinate_descent_tuning`: `78.400 us`
- `torch.compile combo_looped_cd`: `81.056 us`
- Best measured compile/oracle gap: `1.19x`

## Diagnosis

The oracle covers the full Longformer repro scope: it initializes the required
transposed `[768, 2048]` output from `full_2`, scatters the flattened
`bmm_44` values directly into the live cropped attention window when
`view_19` targets survive the `as_strided`/negative-padding crop, preserves the
returned non-contiguous stride, and then reduces the returned layout to the
`[768]` summary. Inductor currently materializes the full indexed
`[2359296]` buffer, views/crops/permutes/clones it, and schedules the sibling
reduction separately because it does not represent `index_put(accumulate=True)`
feeding a cropped/transposed side output as one structured producer. The
actionable fix is `SCATTER_REDUCE`: lower indexed scatter-add into the live
cropped output layout directly and attach the channel-reduction epilogue.

## Commands

```bash
python -m py_compile repros/canonical/sum_5eec4697d8ff/oracle_longformer_scatter_reduce.py
python repros/canonical/sum_5eec4697d8ff/oracle_longformer_scatter_reduce.py --check
python repros/canonical/sum_5eec4697d8ff/oracle_longformer_scatter_reduce.py --bench --warmup 10 --rep 50
```
