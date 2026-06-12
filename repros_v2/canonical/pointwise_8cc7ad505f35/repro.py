"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 8cc7ad505f35
Shape hash: 0105f520
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 1024]"):
        # No stacktrace found for following nodes
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg0_1, 1)
        convert_element_type: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(ne, torch.int32)
        ge: "b8[8, 1024]" = torch.ops.aten.ge.Scalar(arg0_1, 0)
        lt: "b8[8, 1024]" = torch.ops.aten.lt.Scalar(arg0_1, 50265);  arg0_1 = None
        bitwise_and: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        bitwise_and_1: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        return (convert_element_type, unsqueeze)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
