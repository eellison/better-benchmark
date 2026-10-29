"""Curated MM repro. Pattern hash: 9519247d9dd3."""
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
    def forward(self, arg0_1: "bf16[16384, 768]", arg1_1: "bf16[768, 768]", arg2_1: "bf16[384, 64, 512]", arg3_1: "bf16[32, 12, 512, 512]", arg4_1: "f32[32, 12, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_2, arg2_1);  view_2 = arg2_1 = None
        view_3: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        mul: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_3, arg3_1);  view_3 = arg3_1 = None
        convert_element_type: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        mul_1: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, arg4_1);  convert_element_type = arg4_1 = None
        return mul_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
