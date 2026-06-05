# pointwise_c04d168ffc54 - RoPE Layout-Stencil Fusion Gap

## Benchmark Result
- Oracle: 9.82 us
- Compile: 11.46 us
- Ratio: 1.166x
- Status: GOOD (oracle wins)

## Classification
SCHEDULER_FUSION

## Root Cause

The oracle computes the full LLaVA rotary position embedding for both Q and K tensors in **one** Triton kernel. It:
1. Loads position frequencies (f16[64]) once per position
2. Computes cos/sin inline
3. For each of the two matmul inputs (mm0, mm1: f16[512, 4096]), loads the head elements and the "rotated half" (the opposite 64 dims for rotate_half)
4. Applies `x * cos + rotate_half(x) * sin` and stores both outputs with the correct non-contiguous stride (1,32,512,128) -> stride=(2097152, 128, 4096, 1)

Inductor emits **one** large pointwise kernel that fuses everything, but the fusion has suboptimal memory access patterns. The Inductor kernel:
- `triton_poi_fused_add_cat_clone_convert_element_type_cos_expand_iota_mul_neg_permute_sin_slice_unsqueeze_view_0`
- Iterates over the flat output space (1*32*512*128 = 2097152 elements)
- Recomputes cos/sin for every element (frequency * position) rather than loading freq once per (position, freq_idx) pair
- The cat/slice/neg pattern for rotate_half requires conditional indexing per element

The oracle tiles by (position, head_block) and loads the frequency array once per position tile, then broadcasts cos/sin across heads. This saves redundant trig computation.

## Kernel Count
- Inductor: 1 kernel
- Oracle: 1 kernel

The gap is not kernel count but **tiling strategy**: the oracle tiles by (seq_pos, head_block) and reuses cos/sin across heads within a position, while Inductor iterates flat and recomputes trig per element.

## Why Inductor Cannot Do This Today

Inductor's pointwise codegen iterates over the flat output space. When broadcast-dominated ops like cos/sin depend on a subset of dimensions (position and freq_idx, but not head), the computation is redundantly done for each head. The scheduler/codegen does not:
1. Detect that cos/sin only depend on (position, dim%64) 
2. Hoist them into a register-tiled per-position computation
3. Broadcast the result across the head dimension

Additionally, the rotate_half pattern (slice + neg + cat along last dim) creates conditional indexing that the oracle handles with explicit offset arithmetic.

## Config Exploration
- `combo_kernels=True`: no change (already 1 kernel)
- `triton.multi_kernel=3`: no change
- `coordinate_descent_tuning=True`: already enabled

No config resolves this -- it is a tiling/code quality issue within the single fused kernel.

## Fix Direction (Design Doc)

**Enhancement needed**: Broadcast-aware tiling for pointwise kernels with shared sub-dimensional computation.

Two potential approaches:
1. **Tiling heuristic in codegen**: When a pointwise kernel has expensive ops (trig) that depend on fewer dimensions than the output, tile the iteration space so the expensive ops are computed once per their natural dimensionality and broadcast.
2. **RoPE-specific pattern match**: An FX pass that recognizes the RoPE pattern (freq * pos -> cos/sin -> multiply + rotate_half) and lowers it to a specialized Triton template with proper tiling.

Approach 1 is more general but harder. Approach 2 is specific but covers a very common LLM pattern.

**File references**:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (pointwise kernel emission)
- `/tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py` (pattern registration)

**Affected repros**: All LLaVA/Llama-style RoPE computations with multiple Q/K heads sharing position embeddings.

## Source
- Label: torchbench_llava_infer_000
- Pattern: RoPE with freq*pos -> cos/sin broadcast over heads -> rotate_half -> two outputs
