# Local priority oracle measurements batch 2

Timestamp: 2026-06-03T11:25:20+00:00

## sum_sum_sum_7b24a3457260
```bash
python repros/canonical/sum_sum_sum_7b24a3457260/oracle_multi_output_reduction.py --warmup 3 --rep 10 --no-append 
```
```text
oracle_us=26322.113 impl=torch-direct device=cuda warmup=3 rep=10
```
exit_code=0

## sum_sum_cdaed89f373c
```bash
python repros/canonical/sum_sum_cdaed89f373c/oracle_multi_output_reduction.py --warmup 3 --rep 10 --no-append 
```
```text
usage: oracle_multi_output_reduction.py [-h] [--check] [--bench]
                                        [--device DEVICE] [--rtol RTOL]
                                        [--atol ATOL] [--warmup WARMUP]
                                        [--rep REP]
oracle_multi_output_reduction.py: error: unrecognized arguments: --no-append
```
exit_code=2

## sum_sum_6a14a9c9ba88
```bash
python repros/canonical/sum_sum_6a14a9c9ba88/oracle_multi_output_reduction.py --warmup 3 --rep 10 --no-append 
```
```text
usage: oracle_multi_output_reduction.py [-h] [--check] [--bench]
oracle_multi_output_reduction.py: error: unrecognized arguments: --warmup 3 --rep 10 --no-append
```
exit_code=2

## amax_sum_3ed297ef02cd
```bash
python repros/canonical/amax_sum_3ed297ef02cd/oracle_online_softmax.py --warmup 3 --rep 10 --no-append 
```
```text
usage: oracle_online_softmax.py [-h] [--check-only] [--rep REP]
                                [--warmup WARMUP] [--csv CSV] [--no-compile]
oracle_online_softmax.py: error: unrecognized arguments: --no-append
```
exit_code=2

## RETRY sum_sum_cdaed89f373c
```bash
python repros/canonical/sum_sum_cdaed89f373c/oracle_multi_output_reduction.py --bench --warmup 3 --rep 10 
```
```text
Benchmark (sum_sum_cdaed89f373c):
  oracle (fused reduction+epilogue) us=1492.558
  oracle (full including upstream) us=15639.680
  compiled (torch.compile) us=2517.035
  compiled_cd (coordinate_descent_tuning) us=2521.824

  Summary:
    Oracle fused reduction+epilogue: 1492.6 us
    Compiled full graph:             2517.0 us
    Compiled + CD tuning:            2521.8 us
```
exit_code=0

## RETRY sum_sum_6a14a9c9ba88
```bash
python repros/canonical/sum_sum_6a14a9c9ba88/oracle_multi_output_reduction.py --bench 
```
```text

============================================================
Benchmark results for sum_sum_6a14a9c9ba88
============================================================
  Oracle (reduction+pointwise):       2491.8 us
  torch.compile (full repro):         3394.4 us
  torch.compile + coord_descent:      3894.8 us
  Speedup (oracle vs compile):        1.36x
  Speedup (oracle vs coord_descent):  1.56x
============================================================

Note: Oracle measures the fused dual-reduction + post-reduction
pointwise. The scatter_add (max pool backward) is pre-computed.
```
exit_code=0

## RETRY amax_sum_3ed297ef02cd
```bash
python repros/canonical/amax_sum_3ed297ef02cd/oracle_online_softmax.py --warmup 3 --rep 10 --csv investigation_results/measured_oracle_floors.csv 
```
```text
======================================================================
Oracle Online Softmax Benchmark: amax_sum_3ed297ef02cd
Shape: bf16[8192, 262144]
======================================================================

Memory: 8.590 GB total traffic
SOL (3.35 TB/s): 2564.2 us

--- Correctness ---
Correctness check:
  Max absolute difference: 4.768372e-07
  Mean absolute difference: 3.870464e-13
  torch.allclose (atol=1e-2, rtol=1e-2): True
PASSED

--- Benchmark (rep=10, warmup=3) ---

Triton online softmax:
  Median: 1942.9 us
  Effective BW: 4.421 TB/s
  % of SOL: 132.0%

Running torch.compile baseline...
torch.compile:
  Median: 2078.9 us
  Effective BW: 4.132 TB/s
  % of SOL: 123.3%

Speedup (compile / triton): 1.07x

--- Summary ---
  Oracle (Triton online softmax): 1942.9 us
  Baseline (torch.compile): 2078.9 us
  SOL: 2564.2 us
  Oracle achieves 132.0% of SOL

Results appended to investigation_results/measured_oracle_floors.csv
```
exit_code=0

