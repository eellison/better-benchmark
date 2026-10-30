"""Curated MM repro. Pattern hash: 668f35858e4b."""
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
    def forward(self, arg0_1: "bf16[401408, 384]", arg1_1: "bf16[384, 128]", arg2_1: "i64[56]", arg3_1: "f32[128]", arg4_1: "f32[128, 56, 56, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[401408, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8192, 49, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        permute: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        index: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_3, [None, None, arg2_1]);  view_3 = None
        index_1: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index, [None, arg2_1]);  index = arg2_1 = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_1, arg3_1);  arg3_1 = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, 128)
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_1, arg4_1);  arg4_1 = None
        return (index_1, mul, mul_1, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
