# sum_sum_a3ce72227d9d

Gap diagnosis (classification: `SCHEDULER_FUSION`): The oracle consumes the same 14 original inputs as `repro.py` and returns the same `[288]` vector plus `[64, 32, 28, 28]` residual-add slice. It differs from Inductor by fusing the ReLU-mask `where`, channel centering, and two sibling channel reductions across all C=288 BN channels into one split-K Triton reduction, then using those finalized reductions in a slice-limited epilogue for only channels 256:288 while adding the seven live residual slices. Inductor cannot do this today because it schedules the residual slice chain, sibling reductions, dependent BN-backward epilogue, and final slice/add as ordinary graph nodes instead of one full-scope multi-output reduction template that shares the masked/centered producer and sinks the side output to the live channel range. The fix class is `SCHEDULER_FUSION`.

## Queue Position

- Family: `multi_output_reduction_templates`
- Owner: `Codex-cont-multi-a3ce`
- Oracle artifact: `repros/canonical/sum_sum_a3ce72227d9d/oracle_multi_output_reduction.py`

## Oracle State

- Full-scope oracle consumes the same 14 inputs as `repro.py`.
- Returned outputs match `repro.py`: `mul_tensor_8 [288]` and `add_tensor_6 [64, 32, 28, 28]`.
- Correctness command: `python repros/canonical/sum_sum_a3ce72227d9d/oracle_multi_output_reduction.py --check`.
- Correctness result: PASS with matching dtype and stride for both outputs.
- Output 0 max abs `9.765625e-04`, max rel `1.494802e-05`.
- Output 1 max abs `1.907349e-06`, max rel `1.308419e-02`.
- Benchmark command: `python repros/canonical/sum_sum_a3ce72227d9d/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `81.472 us`, `coordinate_descent_tuning=True` compile `137.824 us`, combo-looped-CD compile `137.376 us`.
- Floor verdict: valid full-scope floor candidate; the oracle is faster than both measured compile configs.

## Scope

The timed oracle covers the full compiled repro scope: seven residual channel slices, the mask/select producer, both `[0, 2, 3]` reductions over all 288 channels, the returned `[288]` vector, the dependent BN-backward epilogue for channels 256:288, and the final residual add. It is not a subset reduction or eager timing floor.

## Inductor Closure Path

- Implementation track: multi-output reduction fusion with dependent slice-limited epilogue.
- Candidate hook: let the scheduler form one template for compatible sibling reductions sharing the same producer, then keep the live residual slice chain and epilogue inside the returned channel slice.
- Guardrail: require full repro scope in benchmarking; reduction-only or BN-only kernels are not valid floors for this repro.
