"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: ab0bc2e3c91a
Shape hash: c9bb6ab0
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
    def forward(self, arg0_1: "bf16[128, 768, 7, 7]", arg1_1: "f32[768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[1, 768, 7, 7]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[768]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 768, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 768, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 768, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[768]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9)
        add_1: "f32[768]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_1: "f32[768]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.0001594642002871);  squeeze_1 = None
        mul_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[768]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9)
        add_2: "f32[768]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_1: "bf16[128, 768, 7, 7]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        add_4: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_1, arg5_1);  convert_element_type_1 = arg5_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 768, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 768, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_4, getitem_3)
        mul_7: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_2: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_3: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_2, 0.1)
        mul_9: "f32[768]" = torch.ops.aten.mul.Tensor(arg6_1, 0.9)
        add_6: "f32[768]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_4: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_4, 1.0001594642002871);  squeeze_4 = None
        mul_11: "f32[768]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[768]" = torch.ops.aten.mul.Tensor(arg7_1, 0.9)
        add_7: "f32[768]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_5: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_7: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_8: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_2: "bf16[128, 768, 7, 7]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        unsqueeze_8: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
        unsqueeze_9: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_10: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
        copy_: "f32[768]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = None
        copy__1: "f32[768]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        copy__2: "f32[768]" = torch.ops.aten.copy_.default(arg6_1, add_6);  arg6_1 = add_6 = None
        copy__3: "f32[768]" = torch.ops.aten.copy_.default(arg7_1, add_7);  arg7_1 = add_7 = None
        return (getitem_1, rsqrt, add_4, squeeze_3, convert_element_type_2, unsqueeze_10, copy_, copy__1, copy__2, copy__3)



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
