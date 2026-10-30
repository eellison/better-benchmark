"""Curated MM repro. Pattern hash: a7f1e10d4357."""
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
    def forward(self, arg0_1: "f32[32, 6, 128, 128]", arg1_1: "f32[32, 6, 128, 128]", arg2_1: "f32[32, 6, 128, 128]", arg3_1: "bf16[192, 64, 128]", arg4_1: "bf16[192, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        sum_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [3], True)
        neg: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(arg1_1);  arg1_1 = None
        fma: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, arg0_1);  neg = sum_1 = arg0_1 = None
        convert_element_type: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_2: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  _shape_param_0 = None
        convert_element_type_1: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        add: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None
        bmm: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg3_1, view_2);  arg3_1 = None
        bmm_1: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_2, arg4_1);  view_2 = arg4_1 = None
        view_3: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        view_4: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        permute: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_3, [0, 1, 3, 2]);  view_3 = None
        permute_1: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute, [0, 2, 1, 3]);  permute = None
        view_5: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        clone: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_5, memory_format = torch.contiguous_format);  view_5 = None
        view_6: "bf16[4096, 384]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        permute_2: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone_1: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_8: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        permute_3: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_8, [1, 0])
        return (add, view_6, view_8, permute_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
