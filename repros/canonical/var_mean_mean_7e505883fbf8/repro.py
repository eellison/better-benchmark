"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 7e505883fbf8
Shape hash: d9d8b8eb
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
    def forward(self, arg0_1: "f32[128, 768, 7, 7]", arg1_1: "bf16[128, 768, 7, 7]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        add: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 768, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 768, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[768]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9)
        add_2: "f32[768]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[768]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0001594642002871);  squeeze_2 = None
        mul_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[768]" = torch.ops.aten.mul.Tensor(arg3_1, 0.9)
        add_3: "f32[768]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        mean: "f32[128, 768, 1, 1]" = torch.ops.aten.mean.dim(add_4, [-1, -2], True);  add_4 = None
        as_strided: "f32[128, 768, 1, 1]" = torch.ops.aten.as_strided.default(mean, _shape_param_0, _shape_param_1);  mean = _shape_param_0 = _shape_param_1 = None
        view: "f32[128, 768]" = torch.ops.aten.view.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        convert_element_type: "bf16[128, 768]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        unsqueeze_4: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sub_1: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add, unsqueeze_6);  add = unsqueeze_6 = None
        copy_: "f32[768]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = None
        copy__1: "f32[768]" = torch.ops.aten.copy_.default(arg3_1, add_3);  arg3_1 = add_3 = None
        return (squeeze_1, convert_element_type, sub_1, copy_, copy__1)



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
