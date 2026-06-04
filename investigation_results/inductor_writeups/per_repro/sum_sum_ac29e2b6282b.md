# sum_sum_ac29e2b6282b

## Queue Position

- Family: `multi_output_reduction_templates`
- Owner: `Codex-bottom-multi-ac29`
- Oracle artifact: `repros/canonical/sum_sum_ac29e2b6282b/oracle_multi_output_reduction.py`
- Integration status: diagnosis-only; parent queue should leave the main `oracle_path` blank because the historical-best gate is faster than the artifact.

## Oracle State

- Full-scope diagnostic oracle consumes the same 26 inputs as `repro.py`.
- Returned outputs match `repro.py`: `mul_tensor_8 [416]` and `add_tensor_18 [64, 32, 14, 14]`.
- Correctness command: `python repros/canonical/sum_sum_ac29e2b6282b/oracle_multi_output_reduction.py --check`.
- Correctness result: PASS; output 0 max abs `1.220703e-04`, output 1 max abs `2.861023e-06`, both output strides match.
- Benchmark command: `python repros/canonical/sum_sum_ac29e2b6282b/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`.
- Measured timings: oracle `54.432 us`, `coordinate_descent_tuning=True` compile `56.832 us`, combo-looped-CD compile `45.344 us`.
- Historical-best gate: `31.136000528931618 us`.
- Floor verdict: not a true floor; the oracle is slower than the required combo-looped-CD local config and the historical best.

## Gap Diagnosis

The diagnostic oracle fuses the ReLU-mask producer, centered producer, and two sibling channel reductions over all C=416 channels into one split-K Triton reduction, finalizes both sums while writing the returned `[416]` vector, then computes only the live channels 384:416 of the dependent BN-backward epilogue and adds the nineteen residual input slices. Inductor currently schedules the residual slice/add chain, sibling reductions, dependent pointwise epilogue, and final slice/add as ordinary graph nodes, so it cannot share the masked producer and sink the side output to the returned channel range in one template. Classification: `SCHEDULER_FUSION`.

## Inductor Closure Path

- Implementation track: multi-output reduction fusion with dependent slice-limited epilogue.
- Candidate hook: let the scheduler form one template for compatible sibling reductions sharing the same producer, then keep the live residual slice chain and epilogue inside the returned channel slice.
- Guardrail: this artifact is full-scope but slower than the historical best; do not integrate it as an oracle floor unless a later Triton implementation beats both required local configs and `31.136000528931618 us`.
