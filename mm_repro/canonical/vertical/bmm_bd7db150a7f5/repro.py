"""Curated MM repro. Pattern hash: bd7db150a7f5."""
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
    def forward(self, arg0_1: "f32[32, 16, 128, 128]", arg1_1: "f32[32, 16, 128, 128]", arg2_1: "b8[1, 1, 2048, 2048]", arg3_1: "bf16[512, 128, 128]", arg4_1: "bf16[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [3], True)
        neg: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg1_1);  arg1_1 = None
        fma: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, arg0_1);  neg = sum_1 = arg0_1 = None
        convert_element_type: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg2_1, 2, 0, 128);  arg2_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        where: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, convert_element_type, full);  slice_2 = convert_element_type = full = None
        view: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where, _shape_param_0);  where = _shape_param_0 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg3_1, view);  arg3_1 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view, arg4_1);  view = arg4_1 = None
        view_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        view_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        convert_element_type_1: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        permute: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 1, 3, 2]);  convert_element_type_1 = None
        convert_element_type_2: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute, torch.bfloat16);  permute = None
        convert_element_type_3: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        permute_1: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 2, 1, 3]);  convert_element_type_2 = None
        view_3: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        permute_2: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 2, 1, 3]);  convert_element_type_3 = None
        clone: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        clone_1: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_3, memory_format = torch.contiguous_format);  view_3 = None
        view_5: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_1, _shape_param_4);  clone_1 = _shape_param_4 = None
        permute_3: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_5, [1, 0])
        view_6: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        permute_4: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_6, [1, 0])
        return (view_5, permute_3, view_6, permute_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
