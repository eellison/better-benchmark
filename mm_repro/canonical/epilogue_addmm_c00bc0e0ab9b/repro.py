"""Curated MM repro. Pattern hash: c00bc0e0ab9b."""
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
    def forward(self, arg0_1: "bf16[32, 384, 512]", arg1_1: "f32[384, 1]", arg2_1: "bf16[384]", arg3_1: "bf16[16384, 768]", arg4_1: "bf16[768, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        add: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[32, 384, 512]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg2_1, arg3_1, arg4_1);  arg2_1 = arg3_1 = arg4_1 = None
        view: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        view_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(convert_element_type, [0, 2, 1]);  convert_element_type = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute, view);  permute = view = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_2: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_1: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        return (addmm, view_2, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
