"""
Standalone repro captured via capture_hook.
Label: hf_BertForMaskedLM_infer
Pattern hash: af505388b035
Shape hash: 5428f51a
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
    def forward(self, arg0_1: "bf16[16384, 768]", arg1_1: "bf16[768]", arg2_1: "bf16[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_1);  convert_element_type_2 = getitem_1 = None
        add_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg1_1);  mul_3 = arg1_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, arg2_1);  mul_4 = arg2_1 = None
        convert_element_type_3: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_1);  convert_element_type_3 = _shape_param_1 = None
        return view_1



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
