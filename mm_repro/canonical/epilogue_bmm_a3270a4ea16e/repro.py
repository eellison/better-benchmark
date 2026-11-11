"""Curated MM repro. Pattern hash: a3270a4ea16e."""
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
    def forward(self, arg0_1: "bf16[384, 64, 512]", arg1_1: "bf16[384, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        mul: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view, 0.3535533905932738);  view = None
        permute: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul, [0, 1, 3, 2]);  mul = None
        permute_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute, [0, 2, 1, 3]);  permute = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        clone: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_1, memory_format = torch.contiguous_format);  view_1 = None
        view_2: "bf16[16384, 768]" = torch.ops.aten.view.default(clone, _shape_param_2);  _shape_param_2 = None
        sum_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_1: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (clone, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
