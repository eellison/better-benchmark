# var_mean_052df521537f

## Classification: DROPOUT_RESIDUAL_LAYERNORM_PERMUTE_SIDE

## Current Result

- Family: `gpt2_dropout_residual_layernorm_side`
- Oracle path: `repros/canonical/var_mean_052df521537f/oracle_gpt2_dropout_residual_layernorm_side.py`
- Correctness: PASS (stochastic outputs skipped)
- Oracle: `28.22 us`
- `torch.compile coordinate_descent_tuning=True`: `32.22 us`
- Ratio: 1.142
- Best config: `default (cd=True)`: `32.22 us`
- Status: `real_gap`

## Diagnosis

The oracle computes the complete GPT-2 seeded dropout, residual add, population var_mean, affine LayerNorm, final [8192, 768] -> [768, 8192] permute view, and rsqrt/768 side output in one hidden-row Triton kernel.

Inductor generates a single fused kernel (`triton_per_fused_add_div_gt_inductor_lookup_seed_inductor_random_mul_rsqrt_sub_var_mean_view_0`) that handles all operations including the RNG and reduction. The kernel count matches (1 vs 1).

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 32.22 |
| combo+mk=2 | 35.67 |
| combo+mk=3 | 35.84 |
| Oracle | 28.22 |

No config closes the 1.14x gap. Multi-kernel configs make it slightly worse.

## Root cause

Despite matching kernel count, the oracle's hand-tuned kernel achieves better throughput. The likely cause is that the oracle uses a persistent single-pass approach for hidden=768 (fitting the full row in registers), while Inductor's generated code may use Welford iteration with a reduction loop. The oracle also sinks the transposed output store (permute view) directly into the kernel's store plan, while Inductor may handle the permuted output less efficiently.

## Kernel count
- Oracle: 1 kernel (fused dropout + residual + LN + permute + side output)
- Inductor: 1 kernel (fused per-reduction)

## Recommendation

The 14% gap with matched kernel count suggests a codegen efficiency issue. The oracle uses a persistent row approach with BLOCK_H >= 768, avoiding reduction loops. Inductor's Welford-based reduction template may be using a smaller block and iterating. Improving the persistent reduction threshold for hidden=768 or optimizing the transposed store pattern would help.

Note: exact stochastic equality is not established, so this gap measurement compares kernel launch/compute time without verifying bitwise-identical RNG sequences.

File references: `torch/_inductor/choices.py` (persistent reduction threshold), `torch/_inductor/codegen/triton.py` (transposed store codegen).
