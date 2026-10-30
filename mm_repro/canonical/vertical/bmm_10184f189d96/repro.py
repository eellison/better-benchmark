"""Curated MM repro. Pattern hash: 10184f189d96."""
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "bf16[512, 128, 128]", arg2_1: "f32[32, 16, 128, 128]", arg3_1: "b8[1, 1, 2048, 2048]", arg4_1: "bf16[]", arg5_1: "bf16[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  convert_element_type = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        neg: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg2_1);  arg2_1 = None
        fma: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_1: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg3_1, 2, 0, 128);  arg3_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        where: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, convert_element_type_1, arg4_1);  slice_2 = convert_element_type_1 = arg4_1 = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_1, arg5_1);  arg5_1 = None
        view_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        convert_element_type_2: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        permute: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 2, 1, 3]);  convert_element_type_2 = None
        clone: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        permute_1: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        return (view_1, view_4, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
