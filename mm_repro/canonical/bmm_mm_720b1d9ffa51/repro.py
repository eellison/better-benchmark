"""Curated MM repro. Pattern hash: 720b1d9ffa51."""
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[1024, 1024]", arg2_1: "bf16[256, 64, 512]", arg3_1: "f32[16, 16, 512, 512]", arg4_1: "f32[16, 16, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(view, [0, 1, 4, 2, 3]);  view = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute, [0, 1, 4, 3, 2]);  permute = None
        squeeze: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1, 4);  permute_1 = None
        view_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_1);  squeeze = _shape_param_1 = None
        permute_2: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1, [1, 2, 0, 4, 3]);  view_1 = None
        clone: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_2: "bf16[256, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        bmm: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_2, arg2_1);  view_2 = arg2_1 = None
        squeeze_1: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_3);  bmm = _shape_param_3 = None
        convert_element_type: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        mul: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  convert_element_type = arg3_1 = None
        mul_1: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        return mul_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
