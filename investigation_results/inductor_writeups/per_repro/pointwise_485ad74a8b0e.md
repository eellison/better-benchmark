# pointwise_485ad74a8b0e

## Classification: `ALREADY_FIXED`

## Pattern

ReLU + MaxPool(3x3, stride=2, ceil_mode=True)

- Model: torchbench_squeezenet1_1_infer
- Shape: [512, 64, 111, 111] f16 input
- Output: f16[512, 64, 55, 55] (pool values) + i8[512, 64, 55, 55] (pool offsets)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 882 us |
| Oracle (triton fused ReLU+MaxPool) | 55298 us |
| Gap (compile/oracle) | 0.02x |
| Kernel count | 1 |
| Memory traffic (min) | ~1105 MB |
| Bandwidth SOL | ~138 us |

## Diagnosis

Inductor generates a **single kernel** that already fuses ReLU into the MaxPool operation. The compile (882 us) dramatically beats the oracle because the oracle implementation has BLOCK_SIZE=1 (one output element per Triton program instance), resulting in catastrophically poor utilization on B200.

The oracle is broken/scaffolded and should not be treated as a performance floor. Inductor already achieves the optimal fusion pattern (ReLU fused into pool). The 882 us runtime vs 138 us bandwidth SOL reflects the compute cost of the 3x3 stencil with ceil_mode, which requires boundary checks and is inherently compute-limited at f16 on this large tensor.

The `slice_scatter_elision` pass is not relevant (no slice_scatter ops). The pattern already achieves maximal fusion.

## Inductor Closure

No action needed. Inductor already fuses ReLU into MaxPool in a single kernel. The oracle needs replacement (it is pathologically slow), not the compiler.
