"""Curated MM repro. Pattern hash: cf46935c23fc."""
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
    def forward(self, arg0_1: "bf16[512, 128, 80]", arg1_1: "bf16[512, 80, 128]", arg2_1: "bf16[16, 32, 128, 128]", arg3_1: "bf16[512, 128, 80]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[16, 32, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        neg: "f32[16, 32, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_1);  convert_element_type_1 = None
        fma: "f32[16, 32, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_2: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        bmm_1: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(view_1, arg3_1);  arg3_1 = None
        view_2: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        mul_1: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(view_2, 0.334370152488211);  view_2 = None
        permute: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(mul_1, [0, 2, 1, 3]);  mul_1 = None
        clone: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[2048, 2560]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        permute_1: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_4, [1, 0])
        return (view_1, view_4, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
