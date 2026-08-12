"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_infer
Pattern hash: fc28348c6da6
Shape hash: d7517139
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
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24):
        # No stacktrace found for following nodes
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        unsqueeze: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        permute: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
        iota_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_1, 1024);  convert_element_type_1 = None
        pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_1: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul, 1);  mul = None
        permute_1: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute, permute_1);  permute = permute_1 = None
        sin: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_1)
        cos: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_1);  mul_1 = None
        cat: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_2: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand: "f32[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_0);  unsqueeze_2 = _shape_param_0 = None
        clone: "f32[1024, 16, 1024]" = torch.ops.aten.clone.default(expand);  expand = None
        convert_element_type_2: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_3: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3);  convert_element_type_2 = None
        unsqueeze_4: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        view: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_1);  unsqueeze_4 = _shape_param_1 = None
        squeeze: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        convert_element_type_3: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_5: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 3);  convert_element_type_3 = None
        unsqueeze_6: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        view_1: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_6, _shape_param_2);  unsqueeze_6 = _shape_param_2 = None
        squeeze_1: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_1, 0);  view_1 = None
        convert_element_type_4: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_7: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 3);  convert_element_type_4 = None
        unsqueeze_8: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 4);  unsqueeze_7 = None
        view_2: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_8, _shape_param_3);  unsqueeze_8 = _shape_param_3 = None
        squeeze_2: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_2, 0);  view_2 = None
        convert_element_type_5: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_9: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 3);  convert_element_type_5 = None
        unsqueeze_10: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 4);  unsqueeze_9 = None
        view_3: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_10, _shape_param_4);  unsqueeze_10 = _shape_param_4 = None
        squeeze_3: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_3, 0);  view_3 = None
        convert_element_type_6: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_11: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3);  convert_element_type_6 = None
        unsqueeze_12: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        view_4: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_12, _shape_param_5);  unsqueeze_12 = _shape_param_5 = None
        squeeze_4: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_4, 0);  view_4 = None
        convert_element_type_7: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_13: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_7, 3);  convert_element_type_7 = None
        unsqueeze_14: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
        view_5: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_14, _shape_param_6);  unsqueeze_14 = _shape_param_6 = None
        squeeze_5: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_5, 0);  view_5 = None
        convert_element_type_8: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_15: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, 3);  convert_element_type_8 = None
        unsqueeze_16: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 4);  unsqueeze_15 = None
        view_6: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_16, _shape_param_7);  unsqueeze_16 = _shape_param_7 = None
        squeeze_6: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_6, 0);  view_6 = None
        convert_element_type_9: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_17: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_9, 3);  convert_element_type_9 = None
        unsqueeze_18: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
        view_7: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_18, _shape_param_8);  unsqueeze_18 = _shape_param_8 = None
        squeeze_7: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_7, 0);  view_7 = None
        convert_element_type_10: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_19: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, 3);  convert_element_type_10 = None
        unsqueeze_20: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 4);  unsqueeze_19 = None
        view_8: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_20, _shape_param_9);  unsqueeze_20 = _shape_param_9 = None
        squeeze_8: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_8, 0);  view_8 = None
        convert_element_type_11: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_21: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_11, 3);  convert_element_type_11 = None
        unsqueeze_22: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 4);  unsqueeze_21 = None
        view_9: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_22, _shape_param_10);  unsqueeze_22 = _shape_param_10 = None
        squeeze_9: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_9, 0);  view_9 = None
        convert_element_type_12: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_23: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, 3);  convert_element_type_12 = None
        unsqueeze_24: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 4);  unsqueeze_23 = None
        view_10: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_24, _shape_param_11);  unsqueeze_24 = _shape_param_11 = None
        squeeze_10: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_10, 0);  view_10 = None
        convert_element_type_13: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_25: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_13, 3);  convert_element_type_13 = None
        unsqueeze_26: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 4);  unsqueeze_25 = None
        view_11: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_26, _shape_param_12);  unsqueeze_26 = _shape_param_12 = None
        squeeze_11: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_11, 0);  view_11 = None
        convert_element_type_14: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_27: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, 3);  convert_element_type_14 = None
        unsqueeze_28: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 4);  unsqueeze_27 = None
        view_12: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_28, _shape_param_13);  unsqueeze_28 = _shape_param_13 = None
        squeeze_12: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_12, 0);  view_12 = None
        convert_element_type_15: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_29: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_15, 3);  convert_element_type_15 = None
        unsqueeze_30: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 4);  unsqueeze_29 = None
        view_13: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_30, _shape_param_14);  unsqueeze_30 = _shape_param_14 = None
        squeeze_13: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_13, 0);  view_13 = None
        convert_element_type_16: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_31: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, 3);  convert_element_type_16 = None
        unsqueeze_32: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 4);  unsqueeze_31 = None
        view_14: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_32, _shape_param_15);  unsqueeze_32 = _shape_param_15 = None
        squeeze_14: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_14, 0);  view_14 = None
        convert_element_type_17: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_33: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_17, 3);  convert_element_type_17 = None
        unsqueeze_34: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 4);  unsqueeze_33 = None
        view_15: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_34, _shape_param_16);  unsqueeze_34 = _shape_param_16 = None
        squeeze_15: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_15, 0);  view_15 = None
        convert_element_type_18: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_35: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, 3);  convert_element_type_18 = None
        unsqueeze_36: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, 4);  unsqueeze_35 = None
        view_16: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_36, _shape_param_17);  unsqueeze_36 = _shape_param_17 = None
        squeeze_16: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_16, 0);  view_16 = None
        convert_element_type_19: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_37: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_19, 3);  convert_element_type_19 = None
        unsqueeze_38: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 4);  unsqueeze_37 = None
        view_17: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_38, _shape_param_18);  unsqueeze_38 = _shape_param_18 = None
        squeeze_17: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_17, 0);  view_17 = None
        convert_element_type_20: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_39: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, 3);  convert_element_type_20 = None
        unsqueeze_40: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 4);  unsqueeze_39 = None
        view_18: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_40, _shape_param_19);  unsqueeze_40 = _shape_param_19 = None
        squeeze_18: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_18, 0);  view_18 = None
        convert_element_type_21: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_41: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_21, 3);  convert_element_type_21 = None
        unsqueeze_42: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, 4);  unsqueeze_41 = None
        view_19: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_42, _shape_param_20);  unsqueeze_42 = _shape_param_20 = None
        squeeze_19: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_19, 0);  view_19 = None
        convert_element_type_22: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_43: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, 3);  convert_element_type_22 = None
        unsqueeze_44: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 4);  unsqueeze_43 = None
        view_20: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_44, _shape_param_21);  unsqueeze_44 = _shape_param_21 = None
        squeeze_20: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_20, 0);  view_20 = None
        convert_element_type_23: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_45: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_23, 3);  convert_element_type_23 = None
        unsqueeze_46: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 4);  unsqueeze_45 = None
        view_21: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_46, _shape_param_22);  unsqueeze_46 = _shape_param_22 = None
        squeeze_21: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_21, 0);  view_21 = None
        convert_element_type_24: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16)
        unsqueeze_47: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, 3);  convert_element_type_24 = None
        unsqueeze_48: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_47, 4);  unsqueeze_47 = None
        view_22: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_48, _shape_param_23);  unsqueeze_48 = _shape_param_23 = None
        squeeze_22: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_22, 0);  view_22 = None
        convert_element_type_25: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16);  clone = None
        unsqueeze_49: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_25, 3);  convert_element_type_25 = None
        unsqueeze_50: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 4);  unsqueeze_49 = None
        view_23: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_50, _shape_param_24);  unsqueeze_50 = _shape_param_24 = None
        squeeze_23: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_23, 0);  view_23 = None
        return (squeeze, squeeze_1, squeeze_2, squeeze_3, squeeze_4, squeeze_5, squeeze_6, squeeze_7, squeeze_8, squeeze_9, squeeze_10, squeeze_11, squeeze_12, squeeze_13, squeeze_14, squeeze_15, squeeze_16, squeeze_17, squeeze_18, squeeze_19, squeeze_20, squeeze_21, squeeze_22, squeeze_23)



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
