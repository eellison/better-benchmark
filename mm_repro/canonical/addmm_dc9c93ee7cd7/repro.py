"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/AllenaiLongformerBase/full_graph_003.py
Pattern hash: dc9c93ee7cd7
Shape hash: 6710280d
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "f32[768, 768]", arg2_1: "f32[8, 1024, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        permute: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = view = permute = None
        view_1: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        convert_element_type_3: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.5)
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.7071067811865476);  convert_element_type_3 = None
        erf: "f32[8, 1024, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_4: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_5: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return convert_element_type_5



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
