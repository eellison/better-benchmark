"""cuTile port UNSUPPORTED: PyHPC isoneutral stencil.

The Triton reference is a 30-JIT-function, 1600+ line stencil with irregular
subarray/slice writes across seven output tensors keyed to arbitrary offsets
in [204,204,26]. cuTile lacks a masked scatter path with arbitrary write
offsets and pow-2-only tile shapes make the natural 26-deep and 200-wide
iteration boundaries painful — a faithful port is a rewrite, not a translation.
Leaving as a stub while other targets take priority.
"""

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="b115bb1f", BLOCK=128)
def oracle_forward(inputs, **_kwargs):
    raise NotImplementedError(
        "cuTile port unsupported: 30-kernel PyHPC stencil with slice_scatter "
        "output layouts and non-pow2 (26, 200, 204) tile boundaries not "
        "amenable to a straightforward cuTile translation"
    )
