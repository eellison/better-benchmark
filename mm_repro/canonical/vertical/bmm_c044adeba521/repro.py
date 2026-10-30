"""Curated MM repro. Pattern hash: c044adeba521."""
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
    def forward(self, arg0_1: "f32[1, 16, 128, 128]", arg1_1: "f32[1, 16, 128, 128]", arg2_1: "bf16[16, 256, 128]", arg3_1: "bf16[16, 128, 256]", arg4_1: "bf16[1, 128, 1, 32, 1]", arg5_1: "bf16[1, 128, 1, 32, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [3], True)
        neg: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg1_1);  arg1_1 = None
        fma: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, arg0_1);  neg = sum_1 = arg0_1 = None
        convert_element_type: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        div: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type, 16.0);  convert_element_type = None
        view: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div, _shape_param_0);  div = _shape_param_0 = None
        bmm: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg2_1, view);  arg2_1 = None
        bmm_1: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view, arg3_1);  view = arg3_1 = None
        view_1: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        view_2: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        convert_element_type_1: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        permute: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 1, 3, 2]);  convert_element_type_1 = None
        convert_element_type_2: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute, torch.bfloat16);  permute = None
        convert_element_type_3: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        permute_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 2, 1, 3]);  convert_element_type_3 = None
        permute_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 2, 1, 3]);  convert_element_type_2 = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, 64)
        slice_2: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1, 3, 64, 256);  permute_1 = None
        expand: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_3: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        mul: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_3);  view_3 = None
        expand_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg5_1, _shape_param_5);  arg5_1 = _shape_param_5 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_4: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        mul_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_4);  slice_1 = view_4 = None
        return (permute_2, slice_2, mul, mul_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
