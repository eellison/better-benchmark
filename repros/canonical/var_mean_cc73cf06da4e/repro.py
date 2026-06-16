"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: cc73cf06da4e
Shape hash: 0be627fe
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
    def forward(self, arg0_1: "bf16[128, 160, 8, 8]", arg1_1: "f32[160]", arg2_1: "f32[160]", arg3_1: "f32[160]", arg4_1: "f32[160]", arg5_1: "bf16[128, 160, 8, 8]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 160, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 160, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[160]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9)
        add_1: "f32[160]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_1: "f32[160]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.0001220852154804);  squeeze_1 = None
        mul_4: "f32[160]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[160]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9)
        add_2: "f32[160]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        convert_element_type_2: "f32[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        neg: "f32[128, 160, 8, 8]" = torch.ops.aten.neg.default(convert_element_type_2)
        exp: "f32[128, 160, 8, 8]" = torch.ops.aten.exp.default(neg);  neg = None
        add_4: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 160, 8, 8]" = torch.ops.aten.div.Tensor(convert_element_type_2, add_4);  convert_element_type_2 = add_4 = None
        convert_element_type_3: "bf16[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        cat: "bf16[128, 320, 8, 8]" = torch.ops.aten.cat.default([arg5_1, convert_element_type_3], 1);  arg5_1 = convert_element_type_3 = None
        copy_: "f32[160]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = None
        copy__1: "f32[160]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        return (getitem_1, rsqrt, cat, copy_, copy__1)



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
