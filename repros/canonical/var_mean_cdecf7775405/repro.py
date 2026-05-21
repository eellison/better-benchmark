"""
Standalone repro captured via capture_hook.
Label: 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_infer_000
Pattern hash: cdecf7775405
Shape hash: e8a99cae
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 2560], f32), T([16, 128, 2560], f32), T([2560], f32), T([2560], f32), S([16, 128, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_11: "f32[2048, 2560]", add_13: "f32[16, 128, 2560]", arg36_1: "f32[2560]", arg37_1: "f32[2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48):
        # No stacktrace found for following nodes
        view_default: "f32[16, 128, 2560]" = torch.ops.aten.view.default(addmm_11, _shape_param_0);  addmm_11 = _shape_param_0 = None
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_13, view_default);  add_13 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg36_1);  mul_tensor = arg36_1 = None
        add_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg37_1);  mul_tensor_1 = arg37_1 = None
        view_default_1: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_3);  _shape_param_3 = None
        view_default_4: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_4);  _shape_param_4 = None
        view_default_5: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_5);  _shape_param_5 = None
        view_default_6: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_6);  _shape_param_6 = None
        view_default_7: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_7);  _shape_param_7 = None
        view_default_8: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_8);  _shape_param_8 = None
        view_default_9: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_9);  _shape_param_9 = None
        view_default_10: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_10);  _shape_param_10 = None
        view_default_11: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_11);  _shape_param_11 = None
        view_default_12: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_12);  _shape_param_12 = None
        view_default_13: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_13);  _shape_param_13 = None
        view_default_14: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_14);  _shape_param_14 = None
        view_default_15: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_15);  _shape_param_15 = None
        view_default_16: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_16);  _shape_param_16 = None
        view_default_17: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_17);  _shape_param_17 = None
        view_default_18: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_18);  _shape_param_18 = None
        view_default_19: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_19);  _shape_param_19 = None
        view_default_20: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_20);  _shape_param_20 = None
        view_default_21: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_21);  _shape_param_21 = None
        view_default_22: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_22);  _shape_param_22 = None
        view_default_23: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_23);  _shape_param_23 = None
        view_default_24: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_24);  _shape_param_24 = None
        view_default_25: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_25);  _shape_param_25 = None
        view_default_26: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_26);  _shape_param_26 = None
        view_default_27: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_27);  _shape_param_27 = None
        view_default_28: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_28);  _shape_param_28 = None
        view_default_29: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_29);  _shape_param_29 = None
        view_default_30: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_30);  _shape_param_30 = None
        view_default_31: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_31);  _shape_param_31 = None
        view_default_32: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_32);  _shape_param_32 = None
        view_default_33: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_33);  _shape_param_33 = None
        view_default_34: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_34);  _shape_param_34 = None
        view_default_35: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_35);  _shape_param_35 = None
        view_default_36: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_36);  _shape_param_36 = None
        view_default_37: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_37);  _shape_param_37 = None
        view_default_38: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_38);  _shape_param_38 = None
        view_default_39: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_39);  _shape_param_39 = None
        view_default_40: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_40);  _shape_param_40 = None
        view_default_41: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_41);  _shape_param_41 = None
        view_default_42: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_42);  _shape_param_42 = None
        view_default_43: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_43);  _shape_param_43 = None
        view_default_44: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_44);  _shape_param_44 = None
        view_default_45: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_45);  _shape_param_45 = None
        view_default_46: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_46);  _shape_param_46 = None
        view_default_47: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_47);  _shape_param_47 = None
        view_default_48: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_48);  add_tensor_2 = _shape_param_48 = None
        return (view_default_1, view_default_2, view_default_3, view_default_4, view_default_5, view_default_6, view_default_7, view_default_8, view_default_9, view_default_10, view_default_11, view_default_12, view_default_13, view_default_14, view_default_15, view_default_16, view_default_17, view_default_18, view_default_19, view_default_20, view_default_21, view_default_22, view_default_23, view_default_24, view_default_25, view_default_26, view_default_27, view_default_28, view_default_29, view_default_30, view_default_31, view_default_32, view_default_33, view_default_34, view_default_35, view_default_36, view_default_37, view_default_38, view_default_39, view_default_40, view_default_41, view_default_42, view_default_43, view_default_44, view_default_45, view_default_46, view_default_47, view_default_48)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
