"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: 1ba47c0de2b0
Shape hash: bdff4e84
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
    def forward(self, arg0_1: "f32[32, 512, 768, 2]", arg1_1: "f32[32, 512, 768]", arg2_1: "b8[32, 512, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        select: "f32[32, 512, 768]" = torch.ops.aten.select.int(arg0_1, 3, 0);  arg0_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg1_1, select);  arg1_1 = select = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, mul);  add = mul = None
        view: "f32[16384, 768]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        permute: "f32[768, 16384]" = torch.ops.aten.permute.default(view, [1, 0])
        sum_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view, [0], True)
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        return (view, permute, view_1)



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
