"""Curated MM repro. Pattern hash: b726ec3b5443."""
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
    def forward(self, arg0_1: "bf16[32, 512, 768]", arg1_1: "bf16[768, 384]", arg2_1: "bf16[384]", arg3_1: "bf16[768, 384]", arg4_1: "bf16[384]", arg5_1: "bf16[768, 384]", arg6_1: "bf16[384]", arg7_1: "bf16[32, 512, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        view_1: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        permute_1: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3: "bf16[16384, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_3);  _shape_param_3 = None
        addmm_1: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg4_1, view_3, arg3_1);  arg4_1 = view_3 = arg3_1 = None
        view_5: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        permute_3: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6: "bf16[16384, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_5);  arg0_1 = _shape_param_5 = None
        addmm_2: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg6_1, view_6, arg5_1);  arg6_1 = view_6 = arg5_1 = None
        view_8: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        permute_5: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(arg7_1, view_1);  arg7_1 = view_1 = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_9: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_7);  clone = _shape_param_7 = None
        return (permute_1, permute_3, permute_5, view_9)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
