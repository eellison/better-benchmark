"""Curated MM repro. Pattern hash: 8dcada01dcb3."""
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
    def forward(self, arg0_1: "bf16[192, 512, 64]", arg1_1: "bf16[192, 64, 512]", arg2_1: "b8[8, 1, 512, 512]", arg3_1: "bf16[]", arg4_1: "i64[73]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        where: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(arg2_1, arg3_1, view);  arg2_1 = arg3_1 = view = None
        convert_element_type: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        amax: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = None
        exp: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg4_1, 31);  arg4_1 = None
        inductor_random: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt, div);  div = None
        mul_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        view_1: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_2);  mul_1 = _shape_param_2 = None
        convert_element_type_1: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        permute: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 2, 1])
        return (where, amax, sum_1, gt, convert_element_type_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
