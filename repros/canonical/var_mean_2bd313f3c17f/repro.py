"""
Standalone repro captured via capture_hook.
Label: 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_infer
Pattern hash: 2bd313f3c17f
Shape hash: 7f824027
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
    def forward(self, arg0_1: "bf16[2048, 2560]", arg1_1: "bf16[16, 128, 2560]", arg2_1: "bf16[2560]", arg3_1: "bf16[2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg1_1, view);  arg1_1 = view = None
        convert_element_type: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean[0]
        getitem_1: "f32[16, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type_1: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  _shape_param_2 = None
        view_3: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  _shape_param_3 = None
        view_4: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_4);  _shape_param_4 = None
        view_5: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  _shape_param_5 = None
        view_6: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_6);  _shape_param_6 = None
        view_7: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_7);  _shape_param_7 = None
        view_8: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_8);  _shape_param_8 = None
        view_9: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_9);  _shape_param_9 = None
        view_10: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_10);  _shape_param_10 = None
        view_11: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_11);  _shape_param_11 = None
        view_12: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_12);  _shape_param_12 = None
        view_13: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_13);  _shape_param_13 = None
        view_14: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_14);  _shape_param_14 = None
        view_15: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_15);  _shape_param_15 = None
        view_16: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_16);  _shape_param_16 = None
        view_17: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_17);  _shape_param_17 = None
        view_18: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_18);  _shape_param_18 = None
        view_19: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_19);  _shape_param_19 = None
        view_20: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_20);  _shape_param_20 = None
        view_21: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_21);  _shape_param_21 = None
        view_22: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_22);  _shape_param_22 = None
        view_23: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_23);  _shape_param_23 = None
        view_24: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_24);  _shape_param_24 = None
        view_25: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_25);  _shape_param_25 = None
        view_26: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_26);  _shape_param_26 = None
        view_27: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_27);  _shape_param_27 = None
        view_28: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_28);  _shape_param_28 = None
        view_29: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_29);  _shape_param_29 = None
        view_30: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_30);  _shape_param_30 = None
        view_31: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_31);  _shape_param_31 = None
        view_32: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_32);  _shape_param_32 = None
        view_33: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_33);  _shape_param_33 = None
        view_34: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_34);  _shape_param_34 = None
        view_35: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_35);  _shape_param_35 = None
        view_36: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_36);  _shape_param_36 = None
        view_37: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_37);  _shape_param_37 = None
        view_38: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_38);  _shape_param_38 = None
        view_39: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_39);  _shape_param_39 = None
        view_40: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_40);  _shape_param_40 = None
        view_41: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_41);  _shape_param_41 = None
        view_42: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_42);  _shape_param_42 = None
        view_43: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_43);  _shape_param_43 = None
        view_44: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_44);  _shape_param_44 = None
        view_45: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_45);  _shape_param_45 = None
        view_46: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_46);  _shape_param_46 = None
        view_47: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_47);  _shape_param_47 = None
        view_48: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_48);  _shape_param_48 = None
        return (convert_element_type_1, view_1, view_2, view_3, view_4, view_5, view_6, view_7, view_8, view_9, view_10, view_11, view_12, view_13, view_14, view_15, view_16, view_17, view_18, view_19, view_20, view_21, view_22, view_23, view_24, view_25, view_26, view_27, view_28, view_29, view_30, view_31, view_32, view_33, view_34, view_35, view_36, view_37, view_38, view_39, view_40, view_41, view_42, view_43, view_44, view_45, view_46, view_47, view_48)



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
