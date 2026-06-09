# sum_sum_e6f9af9e33b1

Gap diagnosis (classification: `SCHEDULER_FUSION`): The oracle covers the same computation scope as `repro.py`: it consumes the same 15 inputs, applies the `arg314_1 <= 0` masked producer, shares that producer with `arg312_1 - arg709_1` across both channel reductions, writes the returned `[256]` vector, computes only live epilogue channels 224:256, and adds the eight residual input slices into the returned `[64,32,28,28]` tensor with matching dtype and strides. It differs from Inductor by keeping the residual slice chain, sibling reductions, dependent BN-backward epilogue, and final slice/add in one full-scope Triton schedule. Inductor cannot do this today because the scheduler does not form a multi-output reduction template with a dependent slice-limited epilogue and residual producers inside the same region. The fix class is `SCHEDULER_FUSION`.

## Queue Position

- Family: `multi_output_reduction_templates`
- Owner: `Codex-cont-multi-e6f9`
- Oracle artifact: `repros/canonical/sum_sum_e6f9af9e33b1/oracle_multi_output_reduction.py`

## Oracle State

- Full-scope oracle consumes the same 15 inputs as `repro.py`.
- Returned outputs match `repro.py`: `mul_tensor_8 [256]` and `add_tensor_7 [64, 32, 28, 28]`.
- Correctness command: `python repros/canonical/sum_sum_e6f9af9e33b1/oracle_multi_output_reduction.py --check`.
- Correctness result: PASS; output 0 max abs `7.812500e-03`, max rel `7.997737e-07`, stride match true; output 1 max abs `9.536743e-07`, max rel `8.519221e-03`, stride match true.
- Benchmark command: `python repros/canonical/sum_sum_e6f9af9e33b1/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `80.000 us`, `coordinate_descent_tuning=True` compile `137.760 us`, combo-looped-CD compile `135.776 us`.
- Floor verdict: valid full-scope floor candidate; the oracle is faster than both measured compile configs.

## Inductor Closure Path

- Implementation track: multi-output reduction fusion with dependent slice-limited epilogue.
- Candidate hook: let the scheduler form one template for compatible sibling reductions sharing the same producer, then keep the live residual slice chain and epilogue inside the returned channel slice.
- Guardrail: require full repro scope in benchmarking; reduction-only or BN-only kernels are not valid floors for this repro.
