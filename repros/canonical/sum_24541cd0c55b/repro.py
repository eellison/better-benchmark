"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 24541cd0c55b
Shape hash: 66f209c9
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
    def forward(self, arg0_1: "bf16[256, 512, 512]", arg1_1: "b8[16, 16, 512, 512]", arg2_1: "bf16[16, 16, 512, 512]", arg3_1: "f32[16, 16, 512, 1]", arg4_1: "f32[16, 16, 512, 1]", arg5_1: "b8[16, 16, 512, 1]", arg6_1: "f32[16, 16, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        squeeze: "bf16[16, 16, 512, 512]" = torch.ops.aten.squeeze.dim(view, 4);  view = None
        convert_element_type: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(squeeze, torch.float32);  squeeze = None
        convert_element_type_1: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        mul_2: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(arg2_1, 0.125)
        convert_element_type_2: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float32);  mul_2 = None
        convert_element_type_3: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_3: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1);  convert_element_type_3 = None
        sub: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        mul_4: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub, 0.125);  sub = None
        sub_1: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        where: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(arg5_1, mul_4, sub_1);  arg5_1 = mul_4 = sub_1 = None
        exp: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where);  where = None
        div: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        mul_5: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1, div);  mul_1 = None
        sum_1: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [3], True)
        neg: "f32[16, 16, 512, 512]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[16, 16, 512, 512]" = torch.ops.prims.fma.default(neg, sum_1, mul_5);  neg = sum_1 = mul_5 = None
        convert_element_type_4: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        mul_6: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.125);  convert_element_type_4 = None
        view_1: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.view.default(mul_6, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.permute.default(view_1, [0, 1, 2, 4, 3]);  view_1 = None
        view_2: "bf16[256, 512, 512]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        return (mul_6, view_2)



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
