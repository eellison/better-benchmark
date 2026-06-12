# Adversarial Review Notes

Use this checklist before calling the benchmark corpus clean.

## Worker Queue Recovery

- A repro that raises a normal Python exception must be recorded as failed and must not stop later queued repros.
- A repro that simulates a CUDA/device failure must kill and respawn the persistent worker before later queued repros run.
- GPU lock files must not remain held after worker failure or process death.
- `--output` must include enough failure metadata to resume or debug a partial sweep.
- Successful result entries must keep the existing top-level `{repro.py: {shape: metrics}}` schema for report tools.

## Corpus Validity

- Every canonical repro should have a parseable `meta.json`, `repro.py`, and shape config.
- Shape configs should generate eager-valid inputs before benchmark timing is trusted.
- Index tensors should use bounded generators (`Index(N)`) or permutation generators (`Perm(N)`) where required.
- Full-graph recapture should preserve strides, especially channels-last model inputs.

## Capture / Partitioning

- Horizontal fusion should only join data-dependent regions; independent chains should remain separate.
- Transparent nodes such as `getitem`, views, and tuple plumbing should not hide real data dependencies.
- Re-capture from saved full graphs should avoid full model execution where possible.

## Benchmark Validity

- `memcopy_sol_us` and compiled timing should use the same exclusive timing discipline.
- Multi-worker runs should be compared against single-worker runs for variance before accepting speedup numbers.
- Failed repros should be investigated separately from performance deltas.
