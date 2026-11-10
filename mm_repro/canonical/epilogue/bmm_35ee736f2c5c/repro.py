"""Curated MM repro. Pattern hash: 35ee736f2c5c."""
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
    def forward(self, arg0_1: "bf16[4096, 49, 49]", arg1_1: "bf16[4096, 49, 32]", arg2_1: "bf16[4096, 32, 49]", arg3_1: "bf16[4096, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        bmm: "bf16[4096, 49, 32]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        view_1: "bf16[128, 32, 32, 49]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        view_2: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        permute: "bf16[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2]);  view_1 = None
        mul: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_2, 0.1767766952966369);  view_2 = None
        cat: "bf16[384, 32, 49, 32]" = torch.ops.aten.cat.default([mul, permute, view]);  mul = permute = view = None
        view_3: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        permute_1: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.permute.default(view_3, [1, 3, 0, 2, 4]);  view_3 = None
        clone: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_4: "bf16[128, 49, 3072]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_5: "bf16[6272, 3072]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_2: "bf16[3072, 6272]" = torch.ops.aten.permute.default(view_5, [1, 0])
        sum_1: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32)
        view_6: "f32[3072]" = torch.ops.aten.view.default(sum_1, _shape_param_6);  sum_1 = _shape_param_6 = None
        convert_element_type: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_1: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view_5, permute_2, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
