"""Curated MM repro. Pattern hash: d378923ad103."""
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
    def forward(self, arg0_1: "bf16[9437184]", arg1_1: "bf16[768, 768]", arg2_1: "f32[1024, 8, 768]", arg3_1: "f32[1024, 8, 768]", arg4_1: "f32[8, 1024, 768]", arg5_1: "f32[768]", arg6_1: "f32[8, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        as_strided: "bf16[96, 1536, 64]" = torch.ops.aten.as_strided.default(arg0_1, _shape_param_0, _shape_param_1, 0);  arg0_1 = _shape_param_0 = _shape_param_1 = None
        constant_pad_nd: "bf16[96, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        view: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_3);  constant_pad_nd = _shape_param_3 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        permute_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute, [1, 0, 2, 3]);  permute = None
        clone: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_2, arg1_1);  view_2 = arg1_1 = None
        view_3: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm, _shape_param_5);  mm = _shape_param_5 = None
        convert_element_type: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        add: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(convert_element_type, arg2_1);  convert_element_type = arg2_1 = None
        add_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        permute_2: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg4_1, permute_2);  arg4_1 = permute_2 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_2, arg5_1);  arg5_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_2, arg6_1);  arg6_1 = None
        return (add_2, mul, mul_1, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
