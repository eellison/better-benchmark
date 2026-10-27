"""
Standalone repro captured via capture_hook.
Label: GPT2ForSequenceClassification
Pattern hash: b207817dee40
Shape hash: b44f31b4
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
    def forward(self, arg0_1: "bf16[8, 12, 1024, 64]", arg1_1: "f32[768, 768]", arg2_1: "f32[768]", arg3_1: "i64[25]", arg4_1: "f32[8, 1024, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768, 3072]", arg8_1: "f32[3072]", arg9_1: "f32[3072, 768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768, 2304]", arg14_1: "f32[2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        convert_element_type: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        addmm: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_1, view_1, convert_element_type);  convert_element_type_1 = view_1 = None
        view_2: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 17)
        inductor_random: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, view_2);  view_2 = None
        mul_1: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  arg5_1 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None
        view_3: "f32[8192, 768]" = torch.ops.aten.view.default(add_2, _shape_param_4);  add_2 = _shape_param_4 = None
        convert_element_type_3: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_4: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_5: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        addmm_1: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_5, convert_element_type_4, convert_element_type_3);  convert_element_type_5 = None
        view_4: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_1, _shape_param_5);  _shape_param_5 = None
        mul_4: "bf16[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_4, 0.5)
        convert_element_type_6: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32)
        pow_1: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 3.0);  convert_element_type_6 = None
        mul_5: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_3: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_4, mul_5);  view_4 = mul_5 = None
        mul_6: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.7978845608028654);  add_3 = None
        tanh: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_4: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
        view_5: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_7, _shape_param_6);  mul_7 = _shape_param_6 = None
        convert_element_type_7: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_8: "bf16[8192, 3072]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_9: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        addmm_2: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_9, convert_element_type_8, convert_element_type_7);  convert_element_type_9 = None
        view_6: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_7);  addmm_2 = _shape_param_7 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 18);  arg3_1 = None
        inductor_random_1: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_8, inductor_lookup_seed_1, 'rand');  _shape_param_8 = inductor_lookup_seed_1 = None
        convert_element_type_10: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_10, 0.1);  convert_element_type_10 = None
        mul_8: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_1, view_6);  view_6 = None
        mul_9: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        add_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add, mul_9);  add = mul_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg11_1);  arg11_1 = None
        add_7: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, arg12_1);  mul_11 = arg12_1 = None
        view_7: "f32[8192, 768]" = torch.ops.aten.view.default(add_7, _shape_param_9);  add_7 = _shape_param_9 = None
        convert_element_type_11: "bf16[768, 2304]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_12: "bf16[8192, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        convert_element_type_13: "bf16[2304]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        addmm_3: "bf16[8192, 2304]" = torch.ops.aten.addmm.default(convert_element_type_13, convert_element_type_12, convert_element_type_11);  convert_element_type_13 = None
        view_8: "bf16[8, 1024, 2304]" = torch.ops.aten.view.default(addmm_3, _shape_param_10);  addmm_3 = _shape_param_10 = None
        split = torch.ops.aten.split.Tensor(view_8, 768, 2);  view_8 = None
        getitem_4: "bf16[8, 1024, 768]" = split[0]
        getitem_5: "bf16[8, 1024, 768]" = split[1]
        getitem_6: "bf16[8, 1024, 768]" = split[2];  split = None
        view_9: "bf16[8, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_5, _shape_param_11);  getitem_5 = _shape_param_11 = None
        permute_1: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        view_10: "bf16[8, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_6, _shape_param_12);  getitem_6 = _shape_param_12 = None
        permute_2: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        view_11: "bf16[8, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_4, _shape_param_13);  getitem_4 = _shape_param_13 = None
        permute_3: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        permute_4: "bf16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        permute_5: "bf16[768, 8192]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        div: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_6: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        permute_7: "bf16[3072, 8192]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        permute_8: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        permute_9: "bf16[768, 8192]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        div_1: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        permute_10: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        return (gt, mul_2, addmm_1, gt_1, add_5, mul_10, permute_1, permute_2, permute_3, permute_4, permute_5, div, permute_6, permute_7, permute_8, permute_9, div_1, permute_10)



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
