"""Curated MM repro. Pattern hash: c8fdb860c883."""
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
    def forward(self, arg0_1: "f32[64, 512, 128]", arg1_1: "f32[128]", arg2_1: "f32[128]", arg3_1: "f32[30522]", arg4_1: "f32[128, 30522]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        var_mean = torch.ops.aten.var_mean.correction(arg0_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean[0]
        getitem_1: "f32[64, 512, 1]" = var_mean[1];  var_mean = None
        add: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type: "bf16[30522]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[128, 30522]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_2: "bf16[64, 512, 128]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        constant_pad_nd: "bf16[128, 30528]" = torch.ops.aten.constant_pad_nd.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        full: "bf16[6]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        cat: "bf16[30528]" = torch.ops.aten.cat.default([convert_element_type, full]);  convert_element_type = full = None
        addmm: "bf16[32768, 30528]" = torch.ops.aten.addmm.default(cat, view, constant_pad_nd);  cat = constant_pad_nd = None
        slice_1: "bf16[32768, 30522]" = torch.ops.aten.slice.Tensor(addmm, 1, 0, -6);  addmm = None
        view_1: "bf16[64, 512, 30522]" = torch.ops.aten.view.default(slice_1, _shape_param_3);  slice_1 = _shape_param_3 = None
        return (getitem_1, rsqrt, view, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
