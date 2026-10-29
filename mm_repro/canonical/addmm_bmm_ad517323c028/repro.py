"""Curated MM repro. Pattern hash: ad517323c028."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[768]", arg1_1: "bf16[16384, 768]", arg2_1: "bf16[768, 768]", arg3_1: "f32[768]", arg4_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        addmm: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg2_1 = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        addmm_1: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type, arg1_1, convert_element_type_1);  convert_element_type = arg1_1 = convert_element_type_1 = None
        view_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None
        permute_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_3: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
        clone: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        clone_1: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_5: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_1, _shape_param_3);  clone_1 = _shape_param_3 = None
        bmm: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
        view_6: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_4);  bmm = _shape_param_4 = None
        mul: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.125);  view_6 = None
        convert_element_type_2: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        return convert_element_type_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
