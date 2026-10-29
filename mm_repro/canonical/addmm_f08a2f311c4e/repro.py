"""Curated MM repro. Pattern hash: f08a2f311c4e."""
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
    def forward(self, arg0_1: "bf16[4096, 128]", arg1_1: "bf16[128]", arg2_1: "bf16[4096, 4096]", arg3_1: "bf16[128]", arg4_1: "bf16[128]", arg5_1: "bf16[30000]", arg6_1: "bf16[128, 30000]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        addmm: "bf16[4096, 128]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view: "bf16[8, 512, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        mul: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(view, 0.5)
        pow_1: "bf16[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view, 3.0)
        mul_1: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(view, mul_1);  view = mul_1 = None
        mul_2: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "bf16[8, 512, 128]" = torch.ops.aten.tanh.default(mul_2);  mul_2 = None
        add_1: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_3: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_3, torch.float32);  mul_3 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean[0]
        getitem_1: "f32[8, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_4, arg3_1);  mul_4 = arg3_1 = None
        add_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_5, arg4_1);  mul_5 = arg4_1 = None
        convert_element_type_1: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view_1: "bf16[4096, 128]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        addmm_1: "bf16[4096, 30000]" = torch.ops.aten.addmm.default(arg5_1, view_1, arg6_1);  arg5_1 = view_1 = arg6_1 = None
        view_2: "bf16[8, 512, 30000]" = torch.ops.aten.view.default(addmm_1, _shape_param_2);  _shape_param_2 = None
        convert_element_type_2: "f32[4096, 30000]" = torch.ops.prims.convert_element_type.default(addmm_1, torch.float32);  addmm_1 = None
        return (view_2, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
