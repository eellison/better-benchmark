"""
Standalone repro captured via capture_hook.
Label: GoogleFnet
Pattern hash: e5acca51b7d7
Shape hash: 7d963801
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
    def forward(self, arg0_1: "f32[32, 512, 768, 2]", arg1_1: "f32[32, 512, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[3072, 768]", arg5_1: "f32[3072]", arg6_1: "f32[768, 3072]", arg7_1: "f32[768]", arg8_1: "i64[13]", arg9_1: "f32[768]", arg10_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        select: "f32[32, 512, 768]" = torch.ops.aten.select.int(arg0_1, 3, 0);  arg0_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg1_1, select);  arg1_1 = select = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  arg2_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        view: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, _shape_param_0);  _shape_param_0 = None
        permute: "f32[768, 3072]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg5_1, view, permute);  arg5_1 = permute = None
        view_1: "f32[32, 512, 3072]" = torch.ops.aten.view.default(addmm, _shape_param_1);  _shape_param_1 = None
        mul_2: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_1, 3.0)
        mul_3: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_3: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_1, mul_3);  view_1 = mul_3 = None
        mul_4: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.7978845608028654);  add_3 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_4: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_5: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_2, add_4);  mul_2 = add_4 = None
        view_2: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_5, _shape_param_2);  mul_5 = _shape_param_2 = None
        permute_1: "f32[3072, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg7_1, view_2, permute_1);  arg7_1 = permute_1 = None
        view_3: "f32[32, 512, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg8_1, 7);  arg8_1 = None
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_4, inductor_lookup_seed, 'rand');  _shape_param_4 = inductor_lookup_seed = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, view_3);  view_3 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_7, add_2);  mul_7 = add_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg9_1);  arg9_1 = None
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg10_1);  mul_9 = arg10_1 = None
        convert_element_type: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.complex64)
        div: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        div_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (mul, view, addmm, view_2, gt, mul_8, add_7, convert_element_type, div, div_1)



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
