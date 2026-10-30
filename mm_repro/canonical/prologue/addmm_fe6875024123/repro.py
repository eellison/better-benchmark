"""Curated MM repro. Pattern hash: fe6875024123."""
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
    def forward(self, arg0_1: "f32[16, 128, 1]", arg1_1: "f32[16, 128, 768]", arg2_1: "f32[16, 128, 1]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(arg0_1);  arg0_1 = None
        sub: "f32[16, 128, 768]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sub);  arg3_1 = sub = None
        add: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06);  sqrt = None
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add);  mul = add = None
        add_1: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg4_1);  div = arg4_1 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_2: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[2048, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        addmm: "bf16[2048, 768]" = torch.ops.aten.addmm.default(convert_element_type, view, convert_element_type_1);  convert_element_type = view = convert_element_type_1 = None
        view_2: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_1: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_2: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_3, [0, 2, 1])
        return (view_3, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
