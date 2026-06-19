#!/usr/bin/env python3
"""Smoke test for the post-grad affine reduction extraction prototype."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / "scripts" / "prototype_post_grad_reduction_elim.py"
REPRO = ROOT / "repros" / "canonical" / "sum_sum_sum_d8e421a07bf7" / "repro.py"


@pytest.mark.skipif(
    not REPRO.exists(),
    reason=(
        "Prototype smoke test pinned to the f32 repro sum_sum_sum_d8e421a07bf7, "
        "which was removed by the atomic bf16 corpus flip (b93ba9f4f). The "
        "assertions hard-code that graph's exact node names and f32-specific "
        "reduction_count (131072); the bf16 equivalents have different shapes/"
        "node structure and do not reproduce them, and "
        "prototype_post_grad_reduction_elim.py is exploratory tooling not wired "
        "into any product path. Re-point at a current densenet BN-backward-tail "
        "repro (and refresh the expected output) to re-enable."
    ),
)
def test_densenet_bn_backward_tail_extraction() -> None:
    output = subprocess.check_output(
        [sys.executable, str(PROTOTYPE), str(REPRO)],
        cwd=ROOT,
        text=True,
    )

    assert "candidate: sum_dim_int_list_2 = sum(add_tensor, dims=[0, 2, 3])" in output
    assert "reduction_count: 131072" in output
    assert "sum(slice_tensor)" in output
    assert "sum_dim_int_list" in output
    assert "sum_dim_int_list_1" in output
    assert "sum(arg107_1) - 131072 * arg317_1" in output
    assert "131072 * mul_tensor_1 == sum_dim_int_list" in output


if __name__ == "__main__":
    test_densenet_bn_backward_tail_extraction()
    print("PASS")
