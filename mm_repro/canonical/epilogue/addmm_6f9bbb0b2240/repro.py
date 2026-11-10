"""Curated MM repro. Pattern hash: 6f9bbb0b2240."""
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
    def forward(self, arg0_1: "bf16[1024]", arg1_1: "bf16[6272, 4096]", arg2_1: "bf16[4096, 1024]", arg3_1: "i64[46]", arg4_1: "bf16[128, 49, 1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        addmm: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 45);  arg3_1 = None
        inductor_random: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        lt: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random, 0.8999999985098839);  inductor_random = None
        convert_element_type: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16)
        div: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type, 0.8999999985098839);  convert_element_type = None
        mul: "bf16[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view, div);  view = div = None
        add: "bf16[128, 49, 1024]" = torch.ops.aten.add.Tensor(arg4_1, mul);  arg4_1 = mul = None
        view_1: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        convert_element_type_1: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_1, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = None
        mul_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        add_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        mean: "f32[128, 1024]" = torch.ops.aten.mean.dim(add_2, [1, 2]);  add_2 = None
        convert_element_type_2: "bf16[128, 1024]" = torch.ops.prims.convert_element_type.default(mean, torch.bfloat16);  mean = None
        return (lt, view_1, getitem_1, rsqrt, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
