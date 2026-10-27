"""
Standalone repro captured via capture_hook.
Label: mobilenet_v3_large
Pattern hash: 486f4fb8289d
Shape hash: 844ecbcb
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
    def forward(self, arg0_1: "bf16[32, 960, 7, 7]", arg1_1: "f32[960]", arg2_1: "f32[960]", arg3_1: "f32[960]", arg4_1: "f32[960]", arg5_1: "f32[1280]", arg6_1: "f32[1280, 960]", arg7_1: "f32[1000]", arg8_1: "f32[1000, 1280]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 960, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 960, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze, 0.01);  squeeze = None
        mul_2: "f32[960]" = torch.ops.aten.mul.Tensor(arg1_1, 0.99)
        add_1: "f32[960]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_1: "f32[960]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.0006381620931717);  squeeze_1 = None
        mul_4: "f32[960]" = torch.ops.aten.mul.Tensor(mul_3, 0.01);  mul_3 = None
        mul_5: "f32[960]" = torch.ops.aten.mul.Tensor(arg2_1, 0.99)
        add_2: "f32[960]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        convert_element_type_2: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        add_4: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_2, 3)
        clamp_min: "f32[32, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
        clamp_max: "f32[32, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul_7: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, clamp_max);  convert_element_type_2 = clamp_max = None
        div: "f32[32, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
        convert_element_type_3: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mean: "bf16[32, 960, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_3, [-1, -2], True);  convert_element_type_3 = None
        view: "bf16[32, 960]" = torch.ops.aten.view.default(mean, _shape_param_0);  mean = _shape_param_0 = None
        convert_element_type_4: "bf16[1280]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_5: "bf16[1280, 960]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        permute: "bf16[960, 1280]" = torch.ops.aten.permute.default(convert_element_type_5, [1, 0]);  convert_element_type_5 = None
        addmm: "bf16[32, 1280]" = torch.ops.aten.addmm.default(convert_element_type_4, view, permute);  convert_element_type_4 = None
        convert_element_type_6: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        add_5: "f32[32, 1280]" = torch.ops.aten.add.Tensor(convert_element_type_6, 3)
        clamp_min_1: "f32[32, 1280]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
        clamp_max_1: "f32[32, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        mul_8: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_6, clamp_max_1);  convert_element_type_6 = clamp_max_1 = None
        div_1: "f32[32, 1280]" = torch.ops.aten.div.Tensor(mul_8, 6);  mul_8 = None
        convert_element_type_7: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 1280]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        lt: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(inductor_random, 0.8);  inductor_random = None
        convert_element_type_8: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16)
        div_2: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_8, 0.8);  convert_element_type_8 = None
        mul_9: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_7, div_2);  convert_element_type_7 = div_2 = None
        convert_element_type_9: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_10: "bf16[1000, 1280]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        permute_1: "bf16[1280, 1000]" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        addmm_1: "bf16[32, 1000]" = torch.ops.aten.addmm.default(convert_element_type_9, mul_9, permute_1);  convert_element_type_9 = None
        permute_2: "bf16[1000, 1280]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_3: "bf16[1280, 960]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        copy_: "f32[960]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = None
        copy__1: "f32[960]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        return (getitem_1, rsqrt, view, addmm, lt, mul_9, addmm_1, permute_2, permute_3, copy_, copy__1)



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
