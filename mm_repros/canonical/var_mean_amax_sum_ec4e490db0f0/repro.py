"""
Standalone repro captured via capture_hook.
Label: XGLMForCausalLM
Pattern hash: ec4e490db0f0
Shape hash: 282a29bf
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
    def forward(self, arg0_1: "f32[32, 128, 1024]", arg1_1: "f32[1024]", arg2_1: "f32[1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024, 1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024, 1024]", arg7_1: "f32[1024]", arg8_1: "f32[1024, 1024]", arg9_1: "f32[32, 1, 128, 128]", arg10_1: "f32[1024]", arg11_1: "f32[1024, 1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024]", arg14_1: "f32[4096]", arg15_1: "f32[4096, 1024]", arg16_1: "f32[1024]", arg17_1: "f32[1024, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22):
        # No stacktrace found for following nodes
        var_mean = torch.ops.aten.var_mean.correction(arg0_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        add: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1)
        mul: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_2: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        permute: "bf16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        mul_2: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        convert_element_type_3: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_4: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        permute_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(convert_element_type_3, view, permute_1);  convert_element_type_3 = None
        view_2: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None
        convert_element_type_5: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_6: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        permute_2: "bf16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        addmm_2: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(convert_element_type_5, view, permute_2);  convert_element_type_5 = None
        view_3: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_2, _shape_param_3);  addmm_2 = _shape_param_3 = None
        view_4: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_2, _shape_param_4);  view_2 = _shape_param_4 = None
        permute_3: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        permute_4: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_2, _shape_param_6);  mul_2 = _shape_param_6 = None
        permute_5: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_7: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_7);  clone = _shape_param_7 = None
        clone_1: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_8: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_8);  clone_1 = _shape_param_8 = None
        clone_2: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_9: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_9);  clone_2 = _shape_param_9 = None
        permute_6: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_7, permute_6)
        view_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_10);  _shape_param_10 = None
        add_2: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_10, arg9_1);  view_10 = arg9_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_2, full);  add_2 = full = None
        view_11: "f32[512, 128, 128]" = torch.ops.aten.view.default(maximum, _shape_param_11);  maximum = _shape_param_11 = None
        amax: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_11, [-1], True)
        sub_1: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_11, amax);  view_11 = None
        exp: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_seeds: "i64[3]" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[512, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_12, inductor_lookup_seed, 'rand');  _shape_param_12 = inductor_lookup_seed = None
        gt: "b8[512, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_3: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(gt, div);  div = None
        mul_4: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        convert_element_type_7: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        bmm_1: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_7, view_9)
        view_12: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_13);  bmm_1 = _shape_param_13 = None
        permute_7: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_3: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_13: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_14);  clone_3 = _shape_param_14 = None
        convert_element_type_8: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_9: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        view_14: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_13, _shape_param_15);  view_13 = _shape_param_15 = None
        permute_8: "bf16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm_3: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(convert_element_type_8, view_14, permute_8);  convert_element_type_8 = None
        view_15: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_3, _shape_param_16);  _shape_param_16 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_17, inductor_lookup_seed_1, 'rand');  _shape_param_17 = inductor_lookup_seed_1 = None
        convert_element_type_10: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_10, 0.1);  convert_element_type_10 = None
        mul_5: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_1, view_15);  view_15 = None
        mul_6: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        add_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(arg0_1, mul_6);  arg0_1 = mul_6 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_3, getitem_3)
        mul_7: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_8: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, arg12_1);  mul_7 = arg12_1 = None
        add_5: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_8, arg13_1);  mul_8 = arg13_1 = None
        convert_element_type_11: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        convert_element_type_12: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        convert_element_type_13: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view_16: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_18);  convert_element_type_13 = _shape_param_18 = None
        permute_9: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm_4: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_16, permute_9);  convert_element_type_11 = None
        view_17: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_4, _shape_param_19);  _shape_param_19 = None
        convert_element_type_14: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        mul_9: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_10: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_6: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_11: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_9, add_6);  mul_9 = add_6 = None
        convert_element_type_15: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        convert_element_type_16: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        convert_element_type_17: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        view_18: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_20);  convert_element_type_15 = _shape_param_20 = None
        permute_10: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        addmm_5: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(convert_element_type_16, view_18, permute_10);  convert_element_type_16 = None
        view_19: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_5, _shape_param_21);  addmm_5 = _shape_param_21 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2);  inductor_seeds = None
        inductor_random_2: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_22, inductor_lookup_seed_2, 'rand');  _shape_param_22 = inductor_lookup_seed_2 = None
        convert_element_type_18: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_2, torch.bfloat16);  inductor_random_2 = None
        gt_2: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_18, 0.1);  convert_element_type_18 = None
        mul_12: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_2, view_19);  view_19 = None
        mul_13: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None
        add_7: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_3, mul_13);  add_3 = mul_13 = None
        permute_11: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_12: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_13: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_14: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_7, [0, 2, 1]);  convert_element_type_7 = None
        permute_15: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_16: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_17: "bf16[512, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        permute_18: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_19: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_20: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (getitem_1, rsqrt, view, bmm, amax, sum_1, gt, view_14, addmm_3, gt_1, getitem_3, rsqrt_1, view_16, addmm_4, view_18, gt_2, add_7, permute_11, permute_12, permute_13, permute_14, permute_15, permute_16, permute_17, permute_18, permute_19, permute_20)



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
