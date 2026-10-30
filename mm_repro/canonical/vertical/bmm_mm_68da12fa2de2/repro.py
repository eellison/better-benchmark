"""Curated MM repro. Pattern hash: 68da12fa2de2."""
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
    def forward(self, arg0_1: "bf16[1024, 128, 128]", arg1_1: "bf16[1024, 128, 32]", arg2_1: "bf16[128, 512]", arg3_1: "f32[256, 128, 512]", arg4_1: "f32[256, 128, 512]", arg5_1: "f32[256, 128, 512]", arg6_1: "f32[256, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[32768, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        mm: "bf16[32768, 512]" = torch.ops.aten.mm.default(view_2, arg2_1);  arg2_1 = None
        permute_1: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_2, [1, 0])
        view_3: "bf16[256, 128, 512]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        add: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(arg3_1, convert_element_type);  arg3_1 = convert_element_type = None
        add_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add, arg4_1);  add = arg4_1 = None
        add_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_1, arg5_1);  add_1 = arg5_1 = None
        mul: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_2, arg6_1);  arg6_1 = None
        return (view_2, permute_1, add_2, mul)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
