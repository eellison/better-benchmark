# pointwise_eaafbb6b8c5a


## Measured Timings
- Oracle: 5.76 us
- Compile (CDT): 5.76 us
- Ratio: 1.00x

Gap diagnosis (classification: BANDWIDTH_BOUND): the full-scope Triton oracle mirrors ATen's int64[32] unstable bitonic value/index sort in one program and returns the same `(sorted_values, sorted_indices)` tensors as the repro, whereas Inductor's compiled region reaches the same single-launch tiny-sort floor through its existing sort path; Inductor cannot materially improve this today because the computation is only 32 int64 keys plus 32 indices and is dominated by launch overhead, so the required Inductor change is BANDWIDTH_BOUND: no compiler optimization is indicated.

Status: `not_true_floor`. The Triton sort oracle is exact and full-scope, but it is not faster than the compiled repro on the required benchmark.

Measured with:

```bash
python repros/canonical/pointwise_eaafbb6b8c5a/oracle_sort32.py --bench --warmup 10 --rep 50
```

Result:

```json
{"repro_id": "pointwise_eaafbb6b8c5a", "oracle_us": 3.97, "compile_us": 3.97, "ratio": 1.0, "status": "AT_FLOOR"}
```
