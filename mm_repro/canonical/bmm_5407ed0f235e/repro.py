"""Curated MM repro. Pattern hash: 5407ed0f235e."""
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
    def forward(self, arg0_1: "bf16[256, 512, 1024]", arg1_1: "bf16[256, 1024, 64]", arg2_1: "bf16[256, 512, 512]", arg3_1: "bf16[256, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute, [2, 0, 1, 4, 3]);  permute = None
        squeeze: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1, 4);  permute_1 = None
        convert_element_type: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze, torch.float32);  squeeze = None
        convert_element_type_1: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.bfloat16);  convert_element_type = None
        bmm_1: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        view_1: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm_1, _shape_param_1);  bmm_1 = _shape_param_1 = None
        permute_2: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1, [0, 1, 2, 4, 3]);  view_1 = None
        permute_3: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2, [2, 0, 1, 4, 3]);  permute_2 = None
        squeeze_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_3, 4);  permute_3 = None
        convert_element_type_2: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        convert_element_type_3: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.bfloat16);  convert_element_type_2 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_3);  convert_element_type_1 = convert_element_type_3 = None
        view_2: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        permute_4: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_2, [0, 1, 4, 2, 3]);  view_2 = None
        clone: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        squeeze_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        return squeeze_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
