"""Curated MM repro. Pattern hash: 79bd6a7de316."""
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
    def forward(self, arg0_1: "bf16[128, 6, 128, 49]", arg1_1: "bf16[128, 6, 49, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        clone: "bf16[128, 6, 49, 128]" = torch.ops.aten.clone.default(arg1_1, memory_format = torch.contiguous_format);  arg1_1 = None
        view: "bf16[768, 49, 128]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        clone_1: "bf16[128, 6, 128, 49]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view_1: "bf16[768, 128, 49]" = torch.ops.aten.view.default(clone_1, _shape_param_1);  clone_1 = _shape_param_1 = None
        constant_pad_nd: "bf16[768, 56, 128]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_2);  view = _shape_param_2 = None
        constant_pad_nd_1: "bf16[768, 128, 56]" = torch.ops.aten.constant_pad_nd.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        bmm: "bf16[768, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        slice_1: "bf16[768, 49, 56]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7)
        slice_2: "bf16[768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_2: "bf16[128, 6, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_4);  slice_2 = _shape_param_4 = None
        mul: "bf16[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_2, 0.08838834764831845)
        convert_element_type: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_1: "f32[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_1: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        abs_1: "f32[128, 6, 49, 49]" = torch.ops.aten.abs.default(convert_element_type)
        ne: "b8[128, 6, 49, 49]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        eq: "b8[128, 6, 49, 49]" = torch.ops.aten.eq.Tensor(convert_element_type, convert_element_type)
        mul_2: "b8[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[128, 6, 49, 49]" = torch.ops.aten.logical_not.default(mul_2);  mul_2 = None
        return (bmm, convert_element_type, mul_1, logical_not)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
