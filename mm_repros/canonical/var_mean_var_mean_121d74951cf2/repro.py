"""
Standalone repro captured via capture_hook.
Label: DistillGPT2
Pattern hash: 121d74951cf2
Shape hash: d2402e98
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32, 12, 512, 64]", arg1_1: "bf16[768]", arg2_1: "bf16[768, 768]", arg3_1: "bf16[32, 512, 768]", arg4_1: "bf16[768]", arg5_1: "bf16[768]", arg6_1: "bf16[3072]", arg7_1: "bf16[768, 3072]", arg8_1: "bf16[768]", arg9_1: "bf16[3072, 768]", arg10_1: "bf16[768]", arg11_1: "bf16[768]", arg12_1: "bf16[2304]", arg13_1: "bf16[768, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[16384, 768]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        addmm: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg1_1, view_1, arg2_1);  arg1_1 = view_1 = arg2_1 = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        add: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_2, arg3_1);  view_2 = arg3_1 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        addmm_1: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg6_1, view_3, arg7_1);  arg6_1 = view_3 = arg7_1 = None
        view_4: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        mul_2: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_4, 0.5)
        pow_1: "bf16[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_4, 3.0)
        mul_3: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_3: "bf16[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_4, mul_3);  view_4 = mul_3 = None
        mul_4: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.7978845608028654);  add_3 = None
        tanh: "bf16[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_4: "bf16[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_5: "bf16[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_2, add_4);  mul_2 = add_4 = None
        view_5: "bf16[16384, 3072]" = torch.ops.aten.view.default(mul_5, _shape_param_5);  mul_5 = _shape_param_5 = None
        addmm_2: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg8_1, view_5, arg9_1);  arg8_1 = view_5 = arg9_1 = None
        view_6: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        add_5: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add, view_6);  add = view_6 = None
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_3);  convert_element_type_2 = getitem_3 = None
        add_6: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, arg10_1);  mul_6 = arg10_1 = None
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_7, arg11_1);  mul_7 = arg11_1 = None
        convert_element_type_3: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        view_7: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_7);  convert_element_type_3 = _shape_param_7 = None
        addmm_3: "bf16[16384, 2304]" = torch.ops.aten.addmm.default(arg12_1, view_7, arg13_1);  arg12_1 = view_7 = arg13_1 = None
        view_8: "bf16[32, 512, 2304]" = torch.ops.aten.view.default(addmm_3, _shape_param_8);  addmm_3 = _shape_param_8 = None
        split = torch.ops.aten.split.Tensor(view_8, 768, 2);  view_8 = None
        getitem_4: "bf16[32, 512, 768]" = split[0]
        getitem_5: "bf16[32, 512, 768]" = split[1]
        getitem_6: "bf16[32, 512, 768]" = split[2];  split = None
        view_9: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_4, _shape_param_9);  getitem_4 = _shape_param_9 = None
        permute_1: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        view_10: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_5, _shape_param_10);  getitem_5 = _shape_param_10 = None
        permute_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        view_11: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_6, _shape_param_11);  getitem_6 = _shape_param_11 = None
        permute_3: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        return (add_5, permute_1, permute_2, permute_3)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
