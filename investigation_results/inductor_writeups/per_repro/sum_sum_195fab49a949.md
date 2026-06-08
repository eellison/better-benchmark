# sum_sum_195fab49a949

## Status

- Family: `bandwidth_bound_dual_reduction`
- Closure status: `bad_oracle`
- Artifact: `repros/canonical/sum_sum_195fab49a949/oracle_row_resident_dual_sum.py`
- Classification: `BAD_ORACLE`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py` and returns a single
`float32[4096, 4096]` output. It implements a row-resident Triton kernel for
ALBERT dual-row-sum with final permuted-view layout.

## Timings

- Oracle: 72.32 us
- torch.compile (default): 67.14 us
- Ratio: 0.928x (compiler is faster)

## Gap Diagnosis

The compiled code is already faster than the oracle (ratio 0.928x). Inductor
already emits a single fused f32 dual-reduction/epilogue kernel for the same
full scope and is within the harness floor. The oracle row-resident approach
does not provide a speedup. Classification: BANDWIDTH_BOUND -- runtime is
dominated by required producer reads, two f32 row reductions, and dense output
store. No action needed.

## Validation

- `oracle_row_resident_dual_sum.py --check`: PASS
- `oracle_row_resident_dual_sum.py --bench`: ratio 0.928x, status BAD_ORACLE
