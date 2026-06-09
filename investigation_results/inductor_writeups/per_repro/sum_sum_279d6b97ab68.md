# sum_sum_279d6b97ab68

## Queue Position

- Family: `multi_output_reduction_templates`
- Owner: `Codex-bottom-multi-279d`
- Oracle artifact: `repros/canonical/sum_sum_279d6b97ab68/oracle_multi_output_reduction.py`

## Oracle State

- Full-scope oracle consumes the same 19 inputs as `repro.py`.
- Returned outputs match `repro.py`: `mul_tensor_8 [640]` and `add_tensor_11 [64, 32, 14, 14]`.
- Correctness command: `python repros/canonical/sum_sum_279d6b97ab68/oracle_multi_output_reduction.py --check`.
- Correctness result: PASS; output 0 max abs `1.953125e-03`, output 1 max abs `1.907349e-06`, both output strides match.
- Benchmark command: `python repros/canonical/sum_sum_279d6b97ab68/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `52.000 us`, `coordinate_descent_tuning=True` compile `66.016 us`, combo-looped-CD compile `59.392 us`.
- Floor verdict: valid full-scope floor candidate; the oracle is faster than both measured compile configs.

## Gap Diagnosis

The oracle fuses the ReLU-mask producer, centered producer, and two sibling channel reductions over all C=640 channels into one split-K Triton reduction, finalizes both sums while writing the returned `[640]` vector, then computes only the live channels 608:640 of the dependent BN-backward epilogue and adds the twelve residual input slices. Inductor currently schedules the residual slice/add chain, sibling reductions, dependent pointwise epilogue, and final slice/add as ordinary graph nodes, so it cannot share the masked producer and sink the side output to the returned channel range in one template. Classification: `SCHEDULER_FUSION`.

## Inductor Closure Path

- Implementation track: multi-output reduction fusion with dependent slice-limited epilogue.
- Candidate hook: let the scheduler form one template for compatible sibling reductions sharing the same producer, then keep the live residual slice chain and epilogue inside the returned channel slice.
- Guardrail: require full repro scope in benchmarking; reduction-only or BN-only kernels are not valid floors for this repro.
