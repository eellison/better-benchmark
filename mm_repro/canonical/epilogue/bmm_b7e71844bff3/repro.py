"""Curated MM repro. Pattern hash: b7e71844bff3."""
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
    def forward(self, arg0_1: "bf16[64, 1024, 64]", arg1_1: "bf16[64, 64, 1024]", arg2_1: "i64[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, _shape_param_0);  _shape_param_0 = None
        convert_element_type: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        amax: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_1, [-1], True)
        sub: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_1, amax);  convert_element_type_1 = None
        exp: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_2: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 35);  arg2_1 = None
        inductor_random: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_3: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_3, 0.1);  convert_element_type_3 = None
        mul: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt, convert_element_type_2);  convert_element_type_2 = None
        mul_1: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        expand: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_1, _shape_param_2);  mul_1 = _shape_param_2 = None
        view_1: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        permute: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (bmm, amax, sum_1, gt, view_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
