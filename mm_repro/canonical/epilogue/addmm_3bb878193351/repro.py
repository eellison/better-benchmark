"""Curated MM repro. Pattern hash: 3bb878193351."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128]", arg1_1: "bf16[32768, 512]", arg2_1: "bf16[512, 128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "bf16[32768, 128]", arg6_1: "f32[128]", arg7_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        mul: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view, arg3_1);  view = arg3_1 = None
        add: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul, arg4_1);  mul = arg4_1 = None
        view_1: "bf16[256, 128, 128]" = torch.ops.aten.view.default(arg5_1, _shape_param_1);  arg5_1 = _shape_param_1 = None
        add_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_1, add);  view_1 = add = None
        mul_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_1, arg6_1);  add_1 = arg6_1 = None
        add_2: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        convert_element_type: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        view_2: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        return (addmm, add_2, view_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
