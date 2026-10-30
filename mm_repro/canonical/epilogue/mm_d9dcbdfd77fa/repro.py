"""Curated MM repro. Pattern hash: d9dcbdfd77fa."""
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
    def forward(self, arg0_1: "bf16[4, 1000]", arg1_1: "bf16[1000, 1024]", arg2_1: "bf16[4, 1024, 7, 7]", arg3_1: "f32[1, 1024, 1, 1]", arg4_1: "f32[1, 1024, 1, 1]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[4, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[4, 1024, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand: "bf16[4, 1024, 7, 7]" = torch.ops.aten.expand.default(view, _shape_param_1);  view = _shape_param_1 = None
        div: "bf16[4, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(arg2_1, arg3_1)
        mul: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = arg4_1 = None
        unsqueeze_1: "f32[1024, 1, 1]" = torch.ops.aten.view.default(arg5_1, _shape_param_2);  arg5_1 = _shape_param_2 = None
        mul_1: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "f32[1024, 1, 1]" = torch.ops.aten.view.default(arg6_1, _shape_param_3);  arg6_1 = _shape_param_3 = None
        add: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[4, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        relu: "bf16[4, 1024, 7, 7]" = torch.ops.aten.relu.default(convert_element_type);  convert_element_type = None
        le: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 1024, 7, 7]" = torch.ops.aten.where.self(le, full, div);  le = full = div = None
        convert_element_type_1: "f32[4, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[1024]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze_6: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_4);  squeeze = _shape_param_4 = None
        convert_element_type_2: "f32[4, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub_1: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_6);  convert_element_type_2 = unsqueeze_6 = None
        mul_2: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub_1);  sub_1 = None
        return (convert_element_type_1, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
