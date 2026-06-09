# pointwise_dca18e3e98e9 — Qwen Head Layout

## Status: BAD_ORACLE (ratio = 0.894x)

## Oracle Description
The oracle computes a Qwen/Llama permute->contiguous clone->view chain by directly materializing a contiguous bf16[2048,2048] output from the bf16[B,H,512,D] input using a shape-specialized Triton head-layout copy kernel with 2D grid indexing.

## Classification: NEW_PATTERN (oracle slower than compile)

## Benchmark Results
- Oracle: 9.09 us
- Compiled: 8.13 us
- Ratio: 0.894x (compile is FASTER)

## Root Cause
Inductor's generic pointwise layout materialization already outperforms this hand-written oracle kernel. The permute(0,2,1,3).clone().view(B*S, H*D) pattern is a simple layout copy that Inductor handles efficiently via its generic codegen — likely with better block sizing or vectorization than the oracle's 2D tiled approach.

## Kernel Count
Both execute 1 kernel. Inductor's auto-tuned config wins on this particular shape.

## Config Exploration
No config changes needed — compile already beats oracle.

## Conclusion
Oracle is suboptimal. Inductor already handles this pattern well. No investigation needed.
