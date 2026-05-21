# PR 9 Spec: Fuse outer-reduction into multi-output pointwise producer

## The Problem

When a pointwise kernel produces a buffer that has two consumers:
1. A downstream op that needs the full buffer (e.g., matmul reads `[768, 16384]`)
2. An outer-dim reduction over that buffer (e.g., `sum(dim=0) → [768]`)

Inductor currently emits 3 kernels:

```
K0 (pointwise):  read input[384,64,512] → mul+permute+clone → write buf0[16384, 768]
K1 (reduction):  read buf0[16384, 768] → partial sums → write workspace[768, 128]
K2 (reduction):  read workspace[768, 128] → write output[768]
```

K0 already iterates every element of buf0. K1 re-reads all 48MB of buf0 just
to sum it. That 48MB round-trip through GMEM is pure waste.

## The FX Graph (from BertForMaskedLM backward)

```python
class Repro(torch.nn.Module):
    def forward(self, bmm_70: "f32[384, 64, 512]"):
        reshape_default = aten.reshape.default(bmm_70, [32, 12, 64, 512])
        mul_scalar = aten.mul.Scalar(reshape_default, 0.3535533905932738)
        permute_default = aten.permute.default(mul_scalar, [0, 1, 3, 2])
        permute_default_1 = aten.permute.default(permute_default, [0, 2, 1, 3])
        reshape_default_1 = aten.reshape.default(permute_default_1, [32, 512, 768])
        clone_default = aten.clone.default(reshape_default_1, memory_format=contiguous_format)
        reshape_default_2 = aten.reshape.default(clone_default, [16384, 768])

        # OUTPUT 1: the permuted buffer (needed by downstream matmul)
        permute_default_2 = aten.permute.default(reshape_default_2, [1, 0])  # [768, 16384]

        # OUTPUT 2: outer-dim reduction (bias gradient)
        sum_dim_int_list = aten.sum.dim_IntList(reshape_default_2, [0], True)  # [1, 768]
        reshape_default_3 = aten.reshape.default(sum_dim_int_list, [768])

        return (permute_default_2, reshape_default_3)
```

Key: `reshape_default_2` has TWO users — `permute` (output 1) and `sum` (output 2).

## Generated Kernels (current)

```
TORCH_LOGS="output_code" python region_017_sum_68c1d76d5051_ea349839.py
```

Produces:
```
K0: triton_poi_fused_clone_mul_permute_view_0
    grid: 2D, iterates [16384, 768]
    reads: arg0_1 (input)
    writes: buf0 = [32, 512, 768] = 48MB

K1: triton_red_fused_clone_mul_permute_sum_view_1
    grid: 1D, xnumel=98304 (768*128), r0_numel=128
    reads: buf0 (48MB RE-READ)
    writes: buf1 = [1, 768, 128] (workspace, 384KB)

K2: triton_red_fused_clone_mul_permute_sum_view_2
    grid: 1D, xnumel=768, r0_numel=128
    reads: buf1
    writes: buf2 = [1, 768] (final output)
```

## Why cooperative_reductions doesn't help

The heuristic in `choices.py:366-385` gates cooperative reduction on
`xhint <= 16`. Here `xnumel=768` (the output of the sum), so it returns
False immediately. Cooperative reductions are designed for near-scalar
outputs (like `tensor.sum() → scalar`), not `sum(dim=0) → [768]`.

## Why the scheduler can't fuse K0+K1

K0 is a pointwise node with iteration `[16384, 768]`.
K1 is a reduction node with iteration `([768*128], [128])` — different domain.

The scheduler's `can_fuse` checks require matching iteration domains for
horizontal fusion, and for vertical fusion the consumer (K1) would need to
read the producer's (K0's) output — but K1 is a split-reduction that
operates on the materialized buffer with a completely different tiling.

## The Fix (what a single kernel would look like)

```python
@triton.jit
def fused_pointwise_plus_reduction(
    in_ptr, out_buf_ptr, sum_out_ptr,
    xnumel, rnumel,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
):
    # Each CTA handles XBLOCK rows
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK

    # Per-CTA accumulator for the sum
    acc = tl.zeros([RBLOCK], dtype=tl.float32)

    for row in range(XBLOCK):
        x = xoffset + row
        cols = tl.arange(0, RBLOCK)

        # Load + compute (mul, permute logic)
        val = tl.load(in_ptr + ...) * 0.3535533905932738

        # Write full output (needed by downstream)
        tl.store(out_buf_ptr + x * rnumel + cols, val)

        # Accumulate sum
        acc += val

    # Emit partial sum via atomic_add
    tl.atomic_add(sum_out_ptr + tl.arange(0, RBLOCK), acc)
```

With 16384/XBLOCK CTAs, each doing one atomic_add of [768] elements at the
end. For XBLOCK=8 → 2048 CTAs → 2048 atomic adds per output element.
Negligible contention on B200.

## What the scheduler would need

1. **Recognize the pattern**: A `SchedulerNode` (pointwise) writes `buf0`,
   and `buf0` is consumed by BOTH:
   - Another node that needs the full buffer (permute/matmul)
   - A split-reduction that sums along the pointwise's outer dim

2. **Emit a combined kernel**: The pointwise kernel gets an extra output —
   a reduction accumulator that uses atomic_add. This means the codegen
   needs to support "pointwise store + reduction store" in the same kernel.

3. **Eliminate K1+K2**: Once the sum is produced by K0, the split-reduction
   nodes (K1, K2) become dead and can be removed.

## Repros

```bash
python output/aten_repros/dynamo_BertForMaskedLM/region_017_sum_68c1d76d5051_ea349839.py
python output/aten_repros/dynamo_BertForMaskedLM/region_018_sum_36490c41a058_a53551b1.py
python output/aten_repros/dynamo_BertForMaskedLM/region_019_sum_61a7becdb42d_33c89a90.py
python output/aten_repros/dynamo_BertForMaskedLM/region_020_sum_e90d302edc1d_bcd9ce40.py
```

All show the same pattern: pointwise → split-reduction (3 kernels where 1 suffices).

## Related code

- Split-reduction decision: `/tmp/pytorch-work/torch/_inductor/ir.py:1346` (`Reduction.num_splits()`)
- Cooperative reduction heuristic: `/tmp/pytorch-work/torch/_inductor/choices.py:366`
- Fusion scoring: `/tmp/pytorch-work/torch/_inductor/scheduler.py` (`can_fuse`, `score_fusion`)
- Hand-written fix prototype: `fixes/fix_weight_grad_sum.py`

## Breadth

This pattern appears in every transformer training backward pass where the
weight gradient sum (`sum(dim=0)` for bias grad) follows a pointwise kernel.
BertForMaskedLM alone has 4+ instances per forward/backward.
