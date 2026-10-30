"""Curated MM repro. Pattern hash: a9705604839a."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32, 1280]", arg1_1: "bf16[1280, 960]", arg2_1: "bf16[32, 960, 7, 7]", arg3_1: "f32[1, 960, 1, 1]", arg4_1: "f32[1, 960, 1, 1]", arg5_1: "f32[960]", arg6_1: "f32[960]", arg7_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[32, 960]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 960, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand: "bf16[32, 960, 7, 7]" = torch.ops.aten.expand.default(view, _shape_param_1);  view = _shape_param_1 = None
        div: "bf16[32, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        convert_element_type: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        sub: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg2_1, arg3_1)
        mul: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = arg4_1 = None
        unsqueeze_1: "f32[960, 1, 1]" = torch.ops.aten.view.default(arg5_1, _shape_param_2);  arg5_1 = _shape_param_2 = None
        mul_1: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "f32[960, 1, 1]" = torch.ops.aten.view.default(arg6_1, _shape_param_3);  arg6_1 = _shape_param_3 = None
        add: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type_1: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        convert_element_type_2: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        le: "b8[32, 960, 7, 7]" = torch.ops.aten.le.Scalar(convert_element_type_2, -3)
        lt: "b8[32, 960, 7, 7]" = torch.ops.aten.lt.Scalar(convert_element_type_2, 3)
        div_1: "f32[32, 960, 7, 7]" = torch.ops.aten.div.Tensor(convert_element_type_2, 3);  convert_element_type_2 = None
        add_1: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_1, 0.5);  div_1 = None
        mul_2: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  add_1 = None
        where: "f32[32, 960, 7, 7]" = torch.ops.aten.where.self(lt, mul_2, convert_element_type);  lt = mul_2 = convert_element_type = None
        where_1: "f32[32, 960, 7, 7]" = torch.ops.aten.where.self(le, arg7_1, where);  le = arg7_1 = where = None
        convert_element_type_3: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        convert_element_type_4: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        squeeze: "f32[960]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze_6: "f32[1, 960, 1, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_4);  squeeze = _shape_param_4 = None
        convert_element_type_5: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub_1: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_6);  convert_element_type_5 = unsqueeze_6 = None
        mul_3: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_4, sub_1);  sub_1 = None
        return (convert_element_type_4, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
