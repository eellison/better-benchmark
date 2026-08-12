"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: e529e567d636
Shape hash: ed3cce87
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
    def forward(self, arg0_1: "bf16[2048, 3072]", arg1_1: "b8[16, 128, 3072]", arg2_1: "bf16[2048, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_1: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        view_1: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_2: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_2: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476)
        erf: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_4: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_2)
        mul_5: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_6);  convert_element_type_2 = mul_6 = None
        add_1: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add_1);  convert_element_type_1 = add_1 = None
        convert_element_type_3: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_2: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_2);  convert_element_type_3 = _shape_param_2 = None
        permute: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_1: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[3072]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type_4: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_5: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (view_2, permute, convert_element_type_5)



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
