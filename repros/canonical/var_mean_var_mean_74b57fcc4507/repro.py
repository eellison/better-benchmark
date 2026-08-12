"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train
Pattern hash: 74b57fcc4507
Shape hash: ce5d17b1
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
    def forward(self, arg0_1: "bf16[128, 58, 28, 28]", arg1_1: "f32[58]", arg2_1: "f32[58]", arg3_1: "f32[58]", arg4_1: "f32[58]", arg5_1: "bf16[128, 58, 28, 28]", arg6_1: "f32[58]", arg7_1: "f32[58]", arg8_1: "f32[58]", arg9_1: "f32[58]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 58, 28, 28]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 58, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 58, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 58, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 58, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 58, 28, 28]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[128, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[58]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9)
        add_1: "f32[58]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_1: "f32[58]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.00000996502277);  squeeze_1 = None
        mul_4: "f32[58]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[58]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9)
        add_2: "f32[58]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[128, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[128, 58, 28, 28]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        relu: "bf16[128, 58, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        convert_element_type_2: "f32[128, 58, 28, 28]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem_2: "f32[1, 58, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 58, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[1, 58, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 58, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1: "f32[128, 58, 28, 28]" = torch.ops.aten.sub.Tensor(arg5_1, getitem_3);  arg5_1 = None
        mul_7: "f32[128, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_2: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3])
        mul_8: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_2, 0.1);  squeeze_2 = None
        mul_9: "f32[58]" = torch.ops.aten.mul.Tensor(arg6_1, 0.9)
        add_5: "f32[58]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_3: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_3, 1.00000996502277);  squeeze_3 = None
        mul_11: "f32[58]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[58]" = torch.ops.aten.mul.Tensor(arg7_1, 0.9)
        add_6: "f32[58]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_5: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_7: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_7: "f32[128, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_3: "bf16[128, 58, 28, 28]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        relu_1: "bf16[128, 58, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        cat: "bf16[128, 116, 28, 28]" = torch.ops.aten.cat.default([relu, relu_1], 1);  relu = relu_1 = None
        view: "bf16[128, 2, 58, 28, 28]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        permute: "bf16[128, 58, 2, 28, 28]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3, 4]);  view = None
        clone: "bf16[128, 58, 2, 28, 28]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[128, 116, 28, 28]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        split = torch.ops.aten.split.Tensor(view_1, 58, 1);  view_1 = None
        getitem_4: "bf16[128, 58, 28, 28]" = split[0]
        getitem_5: "bf16[128, 58, 28, 28]" = split[1];  split = None
        copy_: "f32[58]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = None
        copy__1: "f32[58]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        copy__2: "f32[58]" = torch.ops.aten.copy_.default(arg6_1, add_5);  arg6_1 = add_5 = None
        copy__3: "f32[58]" = torch.ops.aten.copy_.default(arg7_1, add_6);  arg7_1 = add_6 = None
        return (getitem_1, rsqrt, getitem_3, rsqrt_1, getitem_4, getitem_5, copy_, copy__1, copy__2, copy__3)



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
