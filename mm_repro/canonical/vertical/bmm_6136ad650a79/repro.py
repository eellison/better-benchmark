"""Curated MM repro. Pattern hash: 6136ad650a79."""
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
    def forward(self, arg0_1: "bf16[512, 128, 64]", arg1_1: "bf16[512, 64, 128]", arg2_1: "bf16[32, 1, 128, 128]", arg3_1: "bf16[512, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add, full);  add = full = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum, _shape_param_1);  maximum = _shape_param_1 = None
        convert_element_type: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        amax: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [2], True)
        sub: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [2], True)
        div: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        bmm_1: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_1, arg3_1);  convert_element_type_1 = arg3_1 = None
        view_2: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        permute: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[4096, 1024]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        return view_4

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
