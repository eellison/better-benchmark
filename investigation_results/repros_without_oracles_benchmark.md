# Repros Without Oracles - CUDAGraph Benchmark Results

Benchmarked: 2026-06-08
Config: `coordinate_descent_tuning=True`, `combo_kernels=True`
Method: `torch.compile` + CUDAGraph capture, `do_bench(g.replay(), warmup=25, rep=200, return_mode='min')`
GPU: CUDA_VISIBLE_DEVICES=1

## Results (last 32 of 67 repros without oracles)

| repro_id | status | compile_us | error_if_crashed |
|----------|--------|-----------|-----------------|
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

## Summary

- **Total repros benchmarked**: 32
- **OK**: 32 (100%)
- **CRASH**: 0
- **TIMEOUT**: 0
- **Timing range**: 7.1us - 326.4us
- **Median**: ~35.7us
- All repros are `var_mean_*` family (variance + mean computation patterns)
- Notable outliers: var_mean_4ca91616285c (326.4us), var_mean_6a701483fed1 (326.4us), var_mean_3e029dc6a6fc (172.0us), var_mean_262248c8386b (154.5us)
