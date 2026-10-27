"""cuTile port UNSUPPORTED: PyHPC Thomas solver over multi-domain stencils unsupported."""

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="f2c38fe6")
def oracle_forward(inputs):
    raise NotImplementedError(
        "cuTile port unsupported: PyHPC Thomas solver over multi-domain stencils unsupported"
    )
