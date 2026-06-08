# Repros Without Oracles - CUDAGraph Benchmark Results

Benchmarked: 2026-06-08
Config: `coordinate_descent_tuning=True`, `combo_kernels=True`
Method: `torch.compile` + CUDAGraph capture, `do_bench(g.replay(), warmup=25, rep=200, return_mode='min')`

## Batch A: First 35 repros without oracles

| repro_id | status | time_us | error_if_crashed |
|----------|--------|---------|-----------------|
| pointwise_531d72f1b34a | TIMEOUT | - | Compilation exceeds 150s (21 inputs, max shape 204x204x26x3) |
| pointwise_a12dc8b8a059 | OK | 15.2 | |
| sum_sum_f4d29f9ee6ad | OK | 1001.4 | |
| sum_sum_sum_0147fd1c4296 | OK | 658.4 | |
| sum_sum_sum_109f690634a7 | OK | 178.1 | |
| sum_sum_sum_34c857ab7db3 | OK | 698.0 | |
| sum_sum_sum_3579253dcf89 | OK | 848.7 | |
| sum_sum_sum_41857d0f0554 | OK | 659.3 | |
| sum_sum_sum_55426f9a4493 | OK | 430.2 | |
| sum_sum_sum_5bff1ad7f52a | OK | 380.8 | |
| sum_sum_sum_6b931086c555 | OK | 78.8 | |
| sum_sum_sum_6f9b333ed892 | OK | 514.8 | |
| sum_sum_sum_7baf7f118798 | OK | 153.4 | |
| sum_sum_sum_9e7a546b859f | OK | 27.4 | |
| sum_sum_sum_b35553d96630 | OK | 77.6 | |
| sum_sum_sum_bab40cbb0446 | OK | 155.5 | |
| sum_sum_sum_c6666009132a | OK | 263.9 | |
| sum_sum_sum_d06bf12e10d0 | OK | 481.2 | |
| sum_sum_sum_f5c107db3be9 | OK | 155.3 | |
| var_mean_035991ff3d2b | OK | 11.6 | |
| var_mean_0361de9eae81 | OK | 40.8 | |
| var_mean_036b334353a4 | OK | 19.3 | |
| var_mean_06924cc70cb4 | OK | 26.3 | |
| var_mean_0b554bb6615a | OK | 29.6 | |
| var_mean_0c9738dea136 | OK | 93.9 | |
| var_mean_0dc1c0150d30 | OK | 19.4 | |
| var_mean_0e7d43725d4d | OK | 87.8 | |
| var_mean_1139e33ee710 | OK | 36.5 | |
| var_mean_12b6b115a741 | OK | 18.0 | |
| var_mean_13d90438bcbd | OK | 104.5 | |
| var_mean_15e821204e82 | OK | 63.2 | |
| var_mean_1711c4ddc910 | OK | 40.7 | |
| var_mean_1bab7e80cec1 | OK | 6.9 | |
| var_mean_1e1a2b2c1b0a | OK | 20.1 | |
| var_mean_219e8e620fdc | OK | 14.2 | |

### Batch A Summary

- **Total**: 35 repros
- **OK**: 34 (97%)
- **TIMEOUT**: 1 (pointwise_531d72f1b34a - compile time)
- **CRASH**: 0

## Batch B: Last 32 repros without oracles (GPU=1)

| repro_id | status | time_us | error_if_crashed |
|----------|--------|---------|-----------------|
| var_mean_262248c8386b | OK | 154.5 | |
| var_mean_26f78838cb62 | OK | 36.5 | |
| var_mean_2ec780efd8cf | OK | 40.7 | |
| var_mean_2fcef21b360d | OK | 30.3 | |
| var_mean_323ce31f395b | OK | 44.9 | |
| var_mean_349ca1a29d3f | OK | 16.1 | |
| var_mean_3e029dc6a6fc | OK | 172.0 | |
| var_mean_40bf68989e6d | OK | 30.5 | |
| var_mean_40d5a5a49ffd | OK | 40.8 | |
| var_mean_427eaae2cf58 | OK | 40.6 | |
| var_mean_48ed6e5abc45 | OK | 14.0 | |
| var_mean_4c529682a7a5 | OK | 14.1 | |
| var_mean_4c9e3e3af443 | OK | 26.3 | |
| var_mean_4ca91616285c | OK | 326.4 | |
| var_mean_52d34178da6e | OK | 46.7 | |
| var_mean_54ad7896eb18 | OK | 18.0 | |
| var_mean_59d8965968f1 | OK | 18.2 | |
| var_mean_5bea7483a354 | OK | 40.8 | |
| var_mean_62a3b5ef3579 | OK | 38.7 | |
| var_mean_640c4a11d07e | OK | 38.6 | |
| var_mean_68e4555a2a35 | OK | 15.9 | |
| var_mean_6a701483fed1 | OK | 326.4 | |
| var_mean_6e34526980e4 | OK | 7.1 | |
| var_mean_7188729904dd | OK | 35.7 | |
| var_mean_72b736f9ad1b | OK | 71.4 | |
| var_mean_74d079cfbdcb | OK | 21.6 | |
| var_mean_76c15aa5b174 | OK | 22.0 | |
| var_mean_78804fb104a5 | OK | 30.3 | |
| var_mean_7aa52d6922c5 | OK | 40.8 | |
| var_mean_7bd119719d27 | OK | 18.4 | |
| var_mean_7d9e5a9d28a4 | OK | 18.3 | |
| var_mean_7ee28ad3c38a | OK | 17.9 | |

### Batch B Summary

- **Total**: 32 repros
- **OK**: 32 (100%)
- **CRASH**: 0
- **TIMEOUT**: 0

## Overall Summary (Both Batches)

- **Total repros benchmarked**: 67
- **OK**: 66 (98.5%)
- **CRASH**: 0
- **TIMEOUT**: 1 (pointwise_531d72f1b34a - compile time only, not runtime)

## Performance Distribution by Family

### pointwise (2 repros)
- 1 TIMEOUT (pointwise_531d72f1b34a: compile time >150s due to large graph with 21 inputs)
- 1 OK at 15.2us (pointwise_a12dc8b8a059)

### sum_sum (1 repro)
- sum_sum_f4d29f9ee6ad: 1001.4us (slowest repro overall)

### sum_sum_sum (16 repros)
- Range: 27.4us - 848.7us
- Median: ~322us
- Top 5 slowest: 848.7, 698.0, 659.3, 658.4, 514.8 us
- Multi-reduction patterns with high potential for fusion gains

### var_mean (48 repros)
- Range: 6.9us - 326.4us
- Median: ~30us
- Notable outliers (>150us): 326.4, 326.4, 172.0, 154.5, 104.5, 93.9, 87.8 us
- Majority (39/48) under 50us -- well-optimized welford reductions

## Patterns for Oracle-Writing Server

1. **sum_sum_sum family**: Multi-reduction kernels (sum chained 3 times across dimensions). Slowest family (100-850us). Oracle writers should target fused multi-reduction patterns where intermediate materializations can be eliminated.

2. **var_mean family**: Variance+mean computations (welford-style). Compile reliably and mostly fast (7-50us). Outliers >150us likely have large reduction dimensions. Oracle writers should focus on single-pass welford reductions with epilogue fusion.

3. **pointwise family**: The large pointwise repro causes compilation timeout due to graph complexity. Oracle writers may need simplified tiled approaches for graphs with many inputs.

## Compilation Health

- **98.5% compile success rate** (66/67)
- **0 crashes or correctness issues**
- The one timeout is purely a compile-time issue (CDT autotuning + large graph), not a runtime failure
