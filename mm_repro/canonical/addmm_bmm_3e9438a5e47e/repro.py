"""Curated MM repro. Pattern hash: 3e9438a5e47e."""
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
    def forward(self, arg0_1: "bf16[1024, 1024]", arg1_1: "bf16[1024]", arg2_1: "bf16[16384, 1024]", arg3_1: "bf16[1024]", arg4_1: "bf16[16384, 1024]", arg5_1: "bf16[1024, 1024]", arg6_1: "bf16[64, 1, 256, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        addmm: "bf16[16384, 1024]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        mul: "bf16[64, 256, 1024]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        view_1: "bf16[64, 256, 16, 64]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        permute_1: "bf16[64, 16, 256, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "bf16[1024, 256, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        addmm_1: "bf16[16384, 1024]" = torch.ops.aten.addmm.default(arg3_1, arg4_1, arg5_1);  arg3_1 = arg4_1 = arg5_1 = None
        view_4: "bf16[64, 256, 16, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        permute_2: "bf16[64, 16, 256, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone_1: "bf16[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_5: "bf16[1024, 256, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_4);  clone_1 = _shape_param_4 = None
        permute_3: "bf16[1024, 64, 256]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        bmm: "bf16[1024, 256, 256]" = torch.ops.aten.bmm.default(view_2, permute_3);  view_2 = permute_3 = None
        view_6: "bf16[64, 16, 256, 256]" = torch.ops.aten.view.default(bmm, _shape_param_5);  bmm = _shape_param_5 = None
        add: "bf16[64, 16, 256, 256]" = torch.ops.aten.add.Tensor(view_6, arg6_1);  view_6 = arg6_1 = None
        view_7: "bf16[1024, 256, 256]" = torch.ops.aten.view.default(add, _shape_param_6);  add = _shape_param_6 = None
        convert_element_type: "f32[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
