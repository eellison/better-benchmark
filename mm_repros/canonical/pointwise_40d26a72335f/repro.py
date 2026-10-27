"""
Standalone repro captured via capture_hook.
Label: ghostnet_100
Pattern hash: 40d26a72335f
Shape hash: 498edcad
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
    def forward(self, arg0_1: "bf16[512, 1280, 1, 1]", arg1_1: "f32[1000]", arg2_1: "f32[1000, 1280]", _shape_param_0):
        # No stacktrace found for following nodes
        relu: "bf16[512, 1280, 1, 1]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        convert_element_type: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[1000, 1280]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        permute: "bf16[1280, 1000]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        view: "bf16[512, 1280]" = torch.ops.aten.view.default(relu, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[512, 1000]" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        permute_1: "bf16[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        le: "b8[512, 1280, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (view, addmm, permute_1, le)



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
