"""Curated MM repro. Pattern hash: b50b73471d69."""
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
    def forward(self, arg0_1: "bf16[768]", arg1_1: "bf16[16384, 768]", arg2_1: "bf16[768, 768]", arg3_1: "i64[37]", arg4_1: "f32[32, 512, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        addmm: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 0);  arg3_1 = None
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.1);  convert_element_type = None
        mul: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, view);  gt = view = None
        mul_1: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  arg5_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        view_1: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        div: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (mul_2, add_2, view_1, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
