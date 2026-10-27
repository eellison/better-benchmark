"""
Standalone repro captured via capture_hook.
Label: BlenderbotForCausalLM
Pattern hash: dac97df7f24f
Shape hash: 5b0f6ae9
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
    def forward(self, arg0_1: "bf16[32, 32, 128, 80]", arg1_1: "f32[2560]", arg2_1: "f32[2560, 2560]", arg3_1: "f32[32, 128, 2560]", arg4_1: "f32[2560]", arg5_1: "f32[2560]", arg6_1: "f32[10240]", arg7_1: "f32[10240, 2560]", arg8_1: "f32[2560]", arg9_1: "f32[2560, 10240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        permute: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        convert_element_type: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_1: "bf16[4096, 2560]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[2560, 2560]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[4096, 2560]" = torch.ops.aten.addmm.default(convert_element_type, view_1, permute_1);  convert_element_type = view_1 = None
        view_2: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(addmm, _shape_param_2);  _shape_param_2 = None
        inductor_seeds: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[32, 128, 2560]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[32, 128, 2560]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(gt, view_2);  view_2 = None
        mul_1: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg3_1, mul_1);  arg3_1 = mul_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul_2: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        add_2: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        convert_element_type_3: "bf16[10240]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_4: "bf16[10240, 2560]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_5: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[4096, 2560]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_4);  convert_element_type_5 = _shape_param_4 = None
        permute_2: "bf16[2560, 10240]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[4096, 10240]" = torch.ops.aten.addmm.default(convert_element_type_3, view_3, permute_2);  convert_element_type_3 = None
        view_4: "bf16[32, 128, 10240]" = torch.ops.aten.view.default(addmm_1, _shape_param_5);  _shape_param_5 = None
        convert_element_type_6: "f32[32, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_4: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_5: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[32, 128, 10240]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_3: "f32[32, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_4, add_3);  mul_4 = add_3 = None
        convert_element_type_7: "bf16[32, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_8: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_9: "bf16[2560, 10240]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        view_5: "bf16[4096, 10240]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_6);  convert_element_type_7 = _shape_param_6 = None
        permute_3: "bf16[10240, 2560]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm_2: "bf16[4096, 2560]" = torch.ops.aten.addmm.default(convert_element_type_8, view_5, permute_3);  convert_element_type_8 = None
        view_6: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(addmm_2, _shape_param_7);  addmm_2 = _shape_param_7 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_1: "f32[32, 128, 2560]" = torch.ops.prims.inductor_random.default(_shape_param_8, inductor_lookup_seed_1, 'rand');  _shape_param_8 = inductor_lookup_seed_1 = None
        convert_element_type_10: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[32, 128, 2560]" = torch.ops.aten.gt.Scalar(convert_element_type_10, 0.1);  convert_element_type_10 = None
        mul_7: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(gt_1, view_6);  view_6 = None
        mul_8: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        add_4: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(add, mul_8);  add = mul_8 = None
        permute_4: "bf16[2560, 10240]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_5: "bf16[10240, 2560]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_6: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (addmm, gt, getitem_1, rsqrt, view_3, addmm_1, view_5, gt_1, add_4, permute_4, permute_5, permute_6)



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
