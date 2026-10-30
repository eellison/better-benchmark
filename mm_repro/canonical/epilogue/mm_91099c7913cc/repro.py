"""Curated MM repro. Pattern hash: 91099c7913cc."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 1408]", arg2_1: "f32[1408]", arg3_1: "f32[128, 1408, 7, 7]", arg4_1: "f32[1408]", arg5_1: "f32[1408]", arg6_1: "f32[128, 1408, 7, 7]", arg7_1: "f32[1408]", arg8_1: "f32[1, 1408, 1, 1]", arg9_1: "bf16[128, 1408, 7, 7]", arg10_1: "f32[1, 1408, 1, 1]", arg11_1: "bf16[128, 1408, 7, 7]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        mm: "bf16[128, 1408]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        full: "bf16[180224]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        as_strided_scatter: "bf16[180224]" = torch.ops.aten.as_strided_scatter.default(full, mm, _shape_param_1, _shape_param_2, 0);  full = mm = _shape_param_1 = _shape_param_2 = None
        as_strided: "bf16[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_3, _shape_param_4, 0);  as_strided_scatter = _shape_param_3 = _shape_param_4 = None
        expand: "bf16[128, 1408, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_5);  as_strided = _shape_param_5 = None
        div: "bf16[128, 1408, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        unsqueeze_1: "f32[1408, 1, 1]" = torch.ops.aten.view.default(arg2_1, _shape_param_6);  arg2_1 = _shape_param_6 = None
        mul: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(arg3_1, unsqueeze_1);  arg3_1 = unsqueeze_1 = None
        unsqueeze_3: "f32[1408, 1, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_7);  arg4_1 = _shape_param_7 = None
        add: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul, unsqueeze_3);  mul = unsqueeze_3 = None
        convert_element_type: "bf16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        unsqueeze_5: "f32[1408, 1, 1]" = torch.ops.aten.view.default(arg5_1, _shape_param_8);  arg5_1 = _shape_param_8 = None
        mul_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(arg6_1, unsqueeze_5);  arg6_1 = unsqueeze_5 = None
        unsqueeze_7: "f32[1408, 1, 1]" = torch.ops.aten.view.default(arg7_1, _shape_param_9);  arg7_1 = _shape_param_9 = None
        add_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_7);  mul_1 = unsqueeze_7 = None
        convert_element_type_1: "bf16[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        add_2: "bf16[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        relu: "bf16[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_2);  add_2 = None
        le: "b8[128, 1408, 7, 7]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1408, 7, 7]" = torch.ops.aten.where.self(le, full_1, div);  le = full_1 = div = None
        convert_element_type_2: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_2: "f32[1408]" = torch.ops.aten.squeeze.dims(arg8_1, [0, 2, 3]);  arg8_1 = None
        unsqueeze_10: "f32[1, 1408, 1, 1]" = torch.ops.aten.view.default(squeeze_2, _shape_param_10);  squeeze_2 = _shape_param_10 = None
        convert_element_type_3: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        sub: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_10);  convert_element_type_3 = unsqueeze_10 = None
        mul_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub);  sub = None
        squeeze_3: "f32[1408]" = torch.ops.aten.squeeze.dims(arg10_1, [0, 2, 3]);  arg10_1 = None
        unsqueeze_13: "f32[1, 1408, 1, 1]" = torch.ops.aten.view.default(squeeze_3, _shape_param_11);  squeeze_3 = _shape_param_11 = None
        convert_element_type_4: "f32[128, 1408, 7, 7]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        sub_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_4, unsqueeze_13);  convert_element_type_4 = unsqueeze_13 = None
        mul_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1);  sub_1 = None
        return (convert_element_type_2, mul_2, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
