"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete wide-row bf16 softmax-backward output by splitting each row across column tiles, cooperatively reducing per-row bf16-rounded probability partials, finalizing one row summary per output row, and recomputing the precision-rounded producer in a parallel tiled epilogue, whereas Inductor currently schedules the exp/div/bf16 round-trip producer, the row sum, and the post-reduction fma/cast as one generic wide-row reduction that serializes too much work inside each row program; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K template for extremely wide row reductions with a dependent full-tensor epilogue and precision-preserving producer recompute; the fix is COOPERATIVE_SPLIT_K: add a row-split softmax-backward lowering that emits tiled row-sum partials, finalizes row summaries, and fuses the bf16 output epilogue without full-size f32 intermediates."""
from __future__ import annotations

import sys
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ORACLE_DIR))

from oracle_cooperative_split_k import (  # noqa: E402
    benchmark,
    main,
    make_inputs,
    oracle_cooperative_split_k,
    oracle_forward,
    run_bench,
    run_check,
)

oracle_multi_output_reduction = oracle_cooperative_split_k


if __name__ == "__main__":
    main()
