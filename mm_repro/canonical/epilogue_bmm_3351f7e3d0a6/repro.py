"""Curated MM repro. Pattern hash: 3351f7e3d0a6."""
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
    def forward(self, arg0_1: "bf16[192, 512, 64]", arg1_1: "bf16[192, 64, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default(_shape_param_0, False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        bmm: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        where: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full, full_1, view);  full = full_1 = view = None
        convert_element_type: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_1: "bf16[192, 512, 512]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
