"""Curated MM repro. Pattern hash: afc178eb6009."""
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
    def forward(self, arg0_1: "bf16[768]", arg1_1: "bf16[8192, 768]", arg2_1: "bf16[768, 768]", arg3_1: "f32[4, 2048, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        addmm: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        inductor_seeds: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[4, 2048, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type: "bf16[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[4, 2048, 768]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.1);  convert_element_type = None
        mul: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(gt, view);  gt = view = None
        mul_1: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(arg3_1, mul_1);  arg3_1 = mul_1 = None
        view_1: "f32[8192, 768]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        var_mean = torch.ops.aten.var_mean.correction(view_1, [1], correction = 0, keepdim = True)
        getitem: "f32[8192, 1]" = var_mean[0]
        getitem_1: "f32[8192, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8192, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8192, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(view_1, getitem_1)
        mul_2: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        add_2: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        convert_element_type_1: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        return (addmm, view_1, getitem_1, rsqrt, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
