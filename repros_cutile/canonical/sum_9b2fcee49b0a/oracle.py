"""cuTile port UNSUPPORTED: DeBERTa softmax-backward uses prims.fma (fused MAD)
which rounds differently from cuTile's separate mul+add. 2 out of 50M rows
exceed the rel tolerance (~2% relative error). This is a fundamental rounding
boundary that cuTile cannot match without an fma primitive."""

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="8dadb13a", BLOCK_N=512)
def oracle_forward(inputs, **_kwargs):
    raise NotImplementedError(
        "cuTile port unsupported: prims.fma rounding boundary vs separate mul+add "
        "leaves 2 outlier bf16 values exceeding rel tolerance"
    )
