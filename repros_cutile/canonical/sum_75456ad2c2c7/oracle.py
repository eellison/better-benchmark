"""cuTile port UNSUPPORTED: TrOCR softmax-backward has intrinsic bf16 drift vs eager.

Even the upstream Triton oracle fails eager-comparison with max_diff ~6.7e7 on
random inputs — the fused softmax-backward math on bf16 rounding boundaries
produces values well outside any reasonable tolerance. A mathematically correct
cuTile port shows the same drift (max_diff ~1e6). Keep as stub so the
cuTile-vs-Triton comparison excludes this NUMERICS_FLAG shape.
"""

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="b0ff1d38")
def oracle_forward(inputs):
    raise NotImplementedError(
        "cuTile port unsupported: intrinsic bf16 drift on random inputs; "
        "upstream Triton oracle is also NUMERICS_FLAG (max_diff ~6.7e7)"
    )
