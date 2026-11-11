"""Curated MM repro. Pattern hash: bd4f9a8c69d6."""
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
    def forward(self, arg0_1: "bf16[256, 128, 512]", arg1_1: "bf16[512, 128]", arg2_1: "bf16[128]", arg3_1: "bf16[128]", arg4_1: "bf16[128]", arg5_1: "bf16[512, 128]", arg6_1: "bf16[128]", arg7_1: "bf16[512, 128]", arg8_1: "bf16[128]", arg9_1: "bf16[128]", arg10_1: "bf16[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[32768, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        view_1: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        mul: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_1, arg3_1);  view_1 = arg3_1 = None
        add: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul, arg4_1);  mul = arg4_1 = None
        view_2: "bf16[32768, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_2);  _shape_param_2 = None
        addmm_1: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg6_1, view_2, arg5_1);  arg6_1 = view_2 = arg5_1 = None
        view_4: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        permute_2: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_5: "bf16[32768, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_4);  arg0_1 = _shape_param_4 = None
        addmm_2: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg8_1, view_5, arg7_1);  arg8_1 = view_5 = arg7_1 = None
        view_6: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_2, _shape_param_5);  addmm_2 = _shape_param_5 = None
        mul_1: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_6, arg9_1);  view_6 = arg9_1 = None
        add_1: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_1, arg10_1);  mul_1 = arg10_1 = None
        return (add, clone, add_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
