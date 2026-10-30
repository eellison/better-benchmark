"""Curated MM repro. Pattern hash: d7272a2913dc."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 1280]", arg2_1: "bf16[128, 1280, 7, 7]", arg3_1: "f32[1, 1280, 1, 1]", arg4_1: "f32[1, 1280, 1, 1]", arg5_1: "f32[1280]", arg6_1: "f32[1280]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        mm: "bf16[128, 1280]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        full: "bf16[163840]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        as_strided_scatter: "bf16[163840]" = torch.ops.aten.as_strided_scatter.default(full, mm, _shape_param_1, _shape_param_2, 0);  full = mm = _shape_param_1 = _shape_param_2 = None
        as_strided: "bf16[128, 1280, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_3, _shape_param_4, 0);  as_strided_scatter = _shape_param_3 = _shape_param_4 = None
        expand: "bf16[128, 1280, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_5);  as_strided = _shape_param_5 = None
        div: "bf16[128, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg2_1, arg3_1)
        mul: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = arg4_1 = None
        unsqueeze_1: "f32[1280, 1, 1]" = torch.ops.aten.view.default(arg5_1, _shape_param_6);  arg5_1 = _shape_param_6 = None
        mul_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "f32[1280, 1, 1]" = torch.ops.aten.view.default(arg6_1, _shape_param_7);  arg6_1 = _shape_param_7 = None
        add: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[128, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        le: "b8[128, 1280, 7, 7]" = torch.ops.aten.le.Scalar(convert_element_type, 0.0)
        ge: "b8[128, 1280, 7, 7]" = torch.ops.aten.ge.Scalar(convert_element_type, 6.0);  convert_element_type = None
        bitwise_or: "b8[128, 1280, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le, ge);  le = ge = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1280, 7, 7]" = torch.ops.aten.where.self(bitwise_or, full_1, div);  bitwise_or = full_1 = div = None
        convert_element_type_1: "f32[128, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_2: "f32[1280]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze_6: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(squeeze_2, _shape_param_8);  squeeze_2 = _shape_param_8 = None
        convert_element_type_2: "f32[128, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_6);  convert_element_type_2 = unsqueeze_6 = None
        mul_2: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub_1);  sub_1 = None
        return (convert_element_type_1, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
