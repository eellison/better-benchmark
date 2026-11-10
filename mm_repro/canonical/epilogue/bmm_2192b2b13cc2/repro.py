"""Curated MM repro. Pattern hash: 2192b2b13cc2."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 128]", arg1_1: "bf16[192, 128, 64]", arg2_1: "bf16[192, 64, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        gt: "b8[16, 128]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        unsqueeze: "b8[16, 1, 128]" = torch.ops.aten.unsqueeze.default(gt, 1);  gt = None
        repeat: "b8[16, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        unsqueeze_1: "b8[16, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat, 1);  repeat = None
        eq: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        view: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        div: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view, 8.0);  view = None
        where: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq, full, div);  eq = full = div = None
        convert_element_type: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        expand: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        view_1: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        return (unsqueeze_1, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
