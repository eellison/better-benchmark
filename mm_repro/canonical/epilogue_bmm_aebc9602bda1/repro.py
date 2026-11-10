"""Curated MM repro. Pattern hash: aebc9602bda1."""
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
    def forward(self, arg0_1: "i64[16, 128]", arg1_1: "bf16[192, 128, 64]", arg2_1: "bf16[192, 64, 128]", arg3_1: "i64[61]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        gt: "b8[16, 128]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        unsqueeze: "b8[16, 1, 128]" = torch.ops.aten.unsqueeze.default(gt, 1);  gt = None
        repeat: "b8[16, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        unsqueeze_1: "b8[16, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat, 1);  repeat = None
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        view: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_1);  _shape_param_1 = None
        div: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view, 8.0);  view = None
        eq: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0);  unsqueeze_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq, full, div);  eq = full = div = None
        convert_element_type: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = None
        exp: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 0);  arg3_1 = None
        inductor_random: "f32[16, 12, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        gt_1: "b8[16, 12, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt_1, div_1);  gt_1 = div_1 = None
        mul_1: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        convert_element_type_1: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        expand: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        view_1: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_4);  expand = _shape_param_4 = None
        permute: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (bmm, amax, sum_1, view_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
