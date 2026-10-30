"""Curated MM repro. Pattern hash: c02a1da007ec."""
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
    def forward(self, arg0_1: "f32[256, 4, 128, 128]", arg1_1: "f32[256, 4, 128, 128]", arg2_1: "bf16[1024, 128, 32]", arg3_1: "bf16[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [3], True)
        neg: "f32[256, 4, 128, 128]" = torch.ops.aten.neg.default(arg1_1);  arg1_1 = None
        fma: "f32[256, 4, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, arg0_1);  neg = sum_1 = arg0_1 = None
        convert_element_type: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        bmm: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view, arg2_1);  arg2_1 = None
        view_1: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        mul: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(view_1, 0.4204482076268573);  view_1 = None
        permute: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(mul, [0, 2, 1, 3]);  mul = None
        clone: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "bf16[32768, 128]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        mm: "bf16[32768, 128]" = torch.ops.aten.mm.default(view_3, arg3_1);  arg3_1 = None
        permute_1: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_3, [1, 0])
        view_4: "bf16[256, 128, 128]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type_1: "f32[256, 128, 128]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        return (view, view_3, permute_1, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
