"""Curated MM repro. Pattern hash: 798c593e4ee0."""
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
    def forward(self, arg0_1: "bf16[5, 64, 1, 1]", arg1_1: "bf16[64]", arg2_1: "bf16[64]", arg3_1: "bf16[64]", arg4_1: "bf16[64]", arg5_1: "bf16[64, 5]", arg6_1: "bf16[5]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[5, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 64, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_1: "bf16[64]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9);  arg1_1 = None
        add: "f32[64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        convert_element_type_1: "bf16[64]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3])
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.25);  squeeze_1 = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2, 0.1);  mul_2 = None
        mul_4: "bf16[64]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9);  arg2_1 = None
        add_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        convert_element_type_2: "bf16[64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        sub: "f32[5, 64, 1, 1]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = getitem_1 = None
        add_2: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_5: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze_1: "bf16[64, 1, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        mul_6: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_1);  mul_5 = unsqueeze_1 = None
        unsqueeze_3: "bf16[64, 1, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        add_3: "f32[5, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[5, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[5, 64]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_2);  convert_element_type_3 = _shape_param_2 = None
        addmm: "bf16[5, 5]" = torch.ops.aten.addmm.default(arg6_1, view, arg5_1);  arg6_1 = view = arg5_1 = None
        convert_element_type_4: "f32[5, 5]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        amax: "f32[5, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [1], True)
        sub_1: "f32[5, 5]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        exp: "f32[5, 5]" = torch.ops.aten.exp.default(sub_1)
        sum_1: "f32[5, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[5, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_2: "f32[5, 5]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
        convert_element_type_5: "bf16[5, 5]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        return (convert_element_type_1, convert_element_type_2, addmm, convert_element_type_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
