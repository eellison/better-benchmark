"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer
Pattern hash: a2329fda80e5
Shape hash: 17affd46
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[64, 128, 1024]", arg2_1: "bf16[1024]", arg3_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24):
        # No stacktrace found for following nodes
        view: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg1_1, view);  arg1_1 = view = None
        convert_element_type: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean[0]
        getitem_1: "f32[64, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type_1: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  _shape_param_2 = None
        view_3: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  _shape_param_3 = None
        view_4: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_4);  _shape_param_4 = None
        view_5: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  _shape_param_5 = None
        view_6: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_6);  _shape_param_6 = None
        view_7: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_7);  _shape_param_7 = None
        view_8: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_8);  _shape_param_8 = None
        view_9: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_9);  _shape_param_9 = None
        view_10: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_10);  _shape_param_10 = None
        view_11: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_11);  _shape_param_11 = None
        view_12: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_12);  _shape_param_12 = None
        view_13: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_13);  _shape_param_13 = None
        view_14: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_14);  _shape_param_14 = None
        view_15: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_15);  _shape_param_15 = None
        view_16: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_16);  _shape_param_16 = None
        view_17: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_17);  _shape_param_17 = None
        view_18: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_18);  _shape_param_18 = None
        view_19: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_19);  _shape_param_19 = None
        view_20: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_20);  _shape_param_20 = None
        view_21: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_21);  _shape_param_21 = None
        view_22: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_22);  _shape_param_22 = None
        view_23: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_23);  _shape_param_23 = None
        view_24: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_24);  _shape_param_24 = None
        return (convert_element_type_1, view_1, view_2, view_3, view_4, view_5, view_6, view_7, view_8, view_9, view_10, view_11, view_12, view_13, view_14, view_15, view_16, view_17, view_18, view_19, view_20, view_21, view_22, view_23, view_24)



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
