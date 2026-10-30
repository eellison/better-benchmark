"""Curated MM repro. Pattern hash: afb2114df257."""
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
    def forward(self, arg0_1: "bf16[32, 1000]", arg1_1: "bf16[1000, 1280]", arg2_1: "b8[32, 1280]", arg3_1: "bf16[32, 1280, 7, 7]", arg4_1: "f32[1, 1280, 1, 1]", arg5_1: "f32[1, 1280, 1, 1]", arg6_1: "f32[1280]", arg7_1: "f32[1280]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[32, 1280]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        div: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type, 0.8);  convert_element_type = None
        mul: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(mm, div);  mm = div = None
        unsqueeze_1: "bf16[32, 1280, 1, 1]" = torch.ops.aten.view.default(mul, _shape_param_0);  mul = _shape_param_0 = None
        expand: "bf16[32, 1280, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_1);  unsqueeze_1 = _shape_param_1 = None
        div_1: "bf16[32, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg3_1, arg4_1)
        mul_1: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = arg5_1 = None
        unsqueeze_3: "f32[1280, 1, 1]" = torch.ops.aten.view.default(arg6_1, _shape_param_2);  arg6_1 = _shape_param_2 = None
        mul_2: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        unsqueeze_5: "f32[1280, 1, 1]" = torch.ops.aten.view.default(arg7_1, _shape_param_3);  arg7_1 = _shape_param_3 = None
        add: "f32[32, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_5);  mul_2 = unsqueeze_5 = None
        convert_element_type_1: "bf16[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        relu: "bf16[32, 1280, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        le: "b8[32, 1280, 7, 7]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1280, 7, 7]" = torch.ops.aten.where.self(le, full, div_1);  le = full = div_1 = None
        convert_element_type_2: "f32[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[1280]" = torch.ops.aten.squeeze.dims(arg4_1, [0, 2, 3]);  arg4_1 = None
        unsqueeze_8: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_4);  squeeze = _shape_param_4 = None
        convert_element_type_3: "f32[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub_1: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_8);  convert_element_type_3 = unsqueeze_8 = None
        mul_3: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1);  sub_1 = None
        return (convert_element_type_2, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
