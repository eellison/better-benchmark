"""Curated MM repro. Pattern hash: e2c829b4a8ca."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096]", arg2_1: "bf16[4096, 4096]", arg3_1: "bf16[4096]", arg4_1: "bf16[4096, 4096]", arg5_1: "bf16[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        addmm: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view_1: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        mul: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        clone: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_2: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        addmm_1: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg3_1, arg4_1, arg5_1);  arg3_1 = arg4_1 = arg5_1 = None
        view_4: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None
        permute_2: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_3: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
        mul_1: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_3, 0.3535533905932738);  permute_3 = None
        clone_1: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        view_5: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_1, _shape_param_3);  clone_1 = _shape_param_3 = None
        bmm: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_2, view_5);  view_2 = view_5 = None
        view_6: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_4);  bmm = _shape_param_4 = None
        eq: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_6, -inf)
        logical_not: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        convert_element_type: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        return (logical_not, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
