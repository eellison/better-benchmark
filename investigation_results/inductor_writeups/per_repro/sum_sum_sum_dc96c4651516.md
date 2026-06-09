# sum_sum_sum_dc96c4651516

## Compile: 533.5us, Oracle: 259.0us, Gap: 2.06x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: Streams the Swin window-unpartition/index producer once per row tile, computes the row-local channel reductions, writes the materialized transpose backing buffer, and accumulates the three returned channel reductions as cooperative partials.

## Fix path: Add cooperative split-K multi-output reduction support for producers that must both materialize a side output and return small sibling reductions.

## Status: implemented

## Details

- Model: timm_swin_base_patch4_window7_224_train_001 (3 shapes)
- Pattern: sum x5 reduction (26 ops)
- Oracle: oracle_multi_output_reduction.py
