"""Curated MM repro. Pattern hash: d19265297ab5."""
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
    def forward(self, arg0_1: "bf16[512, 128]", arg1_1: "bf16[128]", arg2_1: "bf16[32768, 512]", arg3_1: "bf16[128]", arg4_1: "bf16[128]", arg5_1: "bf16[512, 128]", arg6_1: "bf16[128]", arg7_1: "bf16[32768, 512]", arg8_1: "bf16[512, 128]", arg9_1: "bf16[128]", arg10_1: "bf16[32768, 512]", arg11_1: "bf16[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        mul: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view, arg3_1);  view = arg3_1 = None
        add: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul, arg4_1);  mul = arg4_1 = None
        view_1: "bf16[32768, 128]" = torch.ops.aten.view.default(add, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[32768, 128]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        addmm_1: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg6_1, arg7_1, arg5_1);  arg6_1 = arg7_1 = arg5_1 = None
        view_4: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        permute_2: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_5: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        addmm_2: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg9_1, arg10_1, arg8_1);  arg9_1 = arg10_1 = arg8_1 = None
        view_6: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_2, _shape_param_5);  addmm_2 = _shape_param_5 = None
        mul_1: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_6, arg11_1);  view_6 = arg11_1 = None
        return (view_1, view_2, view_5, mul_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
