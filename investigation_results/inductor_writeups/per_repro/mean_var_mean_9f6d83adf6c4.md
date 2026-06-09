# mean_var_mean_9f6d83adf6c4


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 33.50 us
- Ratio: N/A

## Classification: CUDA_ERROR

## Pattern

Fused pooled layernorm: CUDA driver error during benchmark.

## Measurements

```
RuntimeError: CUDA driver error: no kernel image is available for execution on the device
```

Both --check and --bench fail with CUDA kernel image error. This likely indicates
the generated kernel uses features not available on this GPU architecture.

## Status: SKIPPED - CUDA compatibility error, cannot benchmark on this hardware
