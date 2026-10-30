"""Curated MM repro. Pattern hash: 795e758627ae."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 640]", arg2_1: "bf16[128, 640, 8, 8]", arg3_1: "f32[1, 640, 1, 1]", arg4_1: "f32[1, 640, 1, 1]", arg5_1: "f32[640]", arg6_1: "f32[640]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        mm: "bf16[128, 640]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        full: "bf16[81920]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        as_strided_scatter: "bf16[81920]" = torch.ops.aten.as_strided_scatter.default(full, mm, _shape_param_1, _shape_param_2, 0);  full = mm = _shape_param_1 = _shape_param_2 = None
        as_strided: "bf16[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_3, _shape_param_4, 0);  as_strided_scatter = _shape_param_3 = _shape_param_4 = None
        expand: "bf16[128, 640, 8, 8]" = torch.ops.aten.expand.default(as_strided, _shape_param_5);  as_strided = _shape_param_5 = None
        div: "bf16[128, 640, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
        convert_element_type: "f32[128, 640, 8, 8]" = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        sub: "f32[128, 640, 8, 8]" = torch.ops.aten.sub.Tensor(arg2_1, arg3_1)
        mul: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = arg4_1 = None
        unsqueeze_1: "f32[640, 1, 1]" = torch.ops.aten.view.default(arg5_1, _shape_param_6);  arg5_1 = _shape_param_6 = None
        mul_1: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "f32[640, 1, 1]" = torch.ops.aten.view.default(arg6_1, _shape_param_7);  arg6_1 = _shape_param_7 = None
        add: "f32[128, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type_1: "bf16[128, 640, 8, 8]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        convert_element_type_2: "f32[128, 640, 8, 8]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        sigmoid: "f32[128, 640, 8, 8]" = torch.ops.aten.sigmoid.default(convert_element_type_2)
        mul_2: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub_1: "f32[128, 640, 8, 8]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_3: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1);  convert_element_type_2 = sub_1 = None
        add_1: "f32[128, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_3, 1);  mul_3 = None
        mul_4: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_2, add_1);  mul_2 = add_1 = None
        convert_element_type_3: "bf16[128, 640, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        convert_element_type_4: "f32[128, 640, 8, 8]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        squeeze_2: "f32[640]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze_6: "f32[1, 640, 1, 1]" = torch.ops.aten.view.default(squeeze_2, _shape_param_8);  squeeze_2 = _shape_param_8 = None
        convert_element_type_5: "f32[128, 640, 8, 8]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub_2: "f32[128, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_6);  convert_element_type_5 = unsqueeze_6 = None
        mul_5: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_4, sub_2);  sub_2 = None
        return (convert_element_type_4, mul_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
