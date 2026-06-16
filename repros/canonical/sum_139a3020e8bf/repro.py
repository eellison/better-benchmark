"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: 139a3020e8bf
Shape hash: 8572c66c
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
    def forward(self, arg0_1: "bf16[25216, 3072]", arg1_1: "bf16[25216, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[128, 197, 3072]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[128, 197, 3072]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[128, 197, 3072]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        convert_element_type_1: "f32[128, 197, 3072]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[128, 197, 3072]" = torch.ops.aten.erf.default(mul);  mul = None
        add: "f32[128, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_1: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_2: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_3: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_2, -0.5);  mul_2 = None
        exp: "f32[128, 197, 3072]" = torch.ops.aten.exp.default(mul_3);  mul_3 = None
        mul_4: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_5: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_4);  convert_element_type_1 = mul_4 = None
        add_1: "f32[128, 197, 3072]" = torch.ops.aten.add.Tensor(mul_1, mul_5);  mul_1 = mul_5 = None
        mul_6: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_2: "bf16[128, 197, 3072]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_2: "bf16[25216, 3072]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        permute: "bf16[3072, 25216]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_1: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[3072]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type_3: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_4: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (view_2, permute, convert_element_type_4)



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
