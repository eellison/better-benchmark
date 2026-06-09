# pointwise_c7938e4bf86d

## Current Result

- Family: `channel_affine_leaky_relu`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_c7938e4bf86d/oracle_channel_affine_leaky_relu.py`
- Correctness: PASS (stochastic output skipped)
- Oracle: `16.00 us`
- `torch.compile coordinate_descent_tuning=True`: `16.22 us`
- Ratio: 1.014 (AT_FLOOR)

## Diagnosis

The oracle computes the full DCGAN BatchNorm-inference affine plus leaky-ReLU pointwise scope with a channel/spatial Triton tile that hoists channel-only mean, variance, weight, bias, and sqrt/reciprocal work once per channel tile row before broadcasting across spatial elements. Inductor emits one flat fused pointwise kernel that repeats the broadcast arithmetic in the output-element loop. Despite the oracle's theoretically better broadcast hoisting, CUDAGraph measurement shows both are at the same floor because the required convolution read and output write dominate. The 1.4% gap is within measurement noise.

- Inductor kernel count: 1
- Oracle kernel count: 1
- Note: output is stochastic (dropout), correctness check skips affected outputs.

## Commands

```bash
python repros/canonical/pointwise_c7938e4bf86d/oracle_channel_affine_leaky_relu.py --check
python repros/canonical/pointwise_c7938e4bf86d/oracle_channel_affine_leaky_relu.py --bench
```
