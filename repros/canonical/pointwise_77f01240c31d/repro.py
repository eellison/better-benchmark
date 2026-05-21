"""
Standalone repro captured via capture_hook.
Label: torchbench_doctr_det_predictor_infer_000
Pattern hash: 77f01240c31d
Shape hash: 25efda26
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
_shapes_config = "(T([64], f32), T([1, 64, 256, 256], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([1, 64, 128, 128], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([1, 64, 64, 64], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([1, 64, 32, 32], f32), T([64], f32), T([64], f32), T([64], f32), S([256, 1]), S([256, 1]), S([256, 1]), S([256, 1]))"

class Repro(torch.nn.Module):
    def forward(self, arg287_1: "f32[64]", convolution_57: "f32[1, 64, 256, 256]", arg288_1: "f32[64]", arg289_1: "f32[64]", arg290_1: "f32[64]", arg292_1: "f32[64]", convolution_58: "f32[1, 64, 128, 128]", arg293_1: "f32[64]", arg294_1: "f32[64]", arg295_1: "f32[64]", arg297_1: "f32[64]", convolution_59: "f32[1, 64, 64, 64]", arg298_1: "f32[64]", arg299_1: "f32[64]", arg300_1: "f32[64]", arg302_1: "f32[64]", convolution_60: "f32[1, 64, 32, 32]", arg303_1: "f32[64]", arg304_1: "f32[64]", arg305_1: "f32[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_default_1);  convolution_57 = unsqueeze_default_1 = None
        add_tensor: "f32[64]" = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_default: "f32[64]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_default_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_default_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        relu_default: "f32[1, 64, 256, 256]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        iota_default: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default: "f32[256]" = torch.ops.prims.convert_element_type.default(iota_default, torch.float32);  iota_default = None
        mul_tensor_3: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        clamp_min_default: "f32[256]" = torch.ops.aten.clamp_min.default(mul_tensor_3, 0.0);  mul_tensor_3 = None
        view_default: "f32[256, 1]" = torch.ops.aten.view.default(clamp_min_default, _shape_param_0);  clamp_min_default = _shape_param_0 = None
        convert_element_type_default_1: "i64[256, 1]" = torch.ops.prims.convert_element_type.default(view_default, torch.int64)
        add_tensor_2: "i64[256, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1)
        clamp_max_default: "i64[256, 1]" = torch.ops.aten.clamp_max.default(add_tensor_2, 255);  add_tensor_2 = None
        iota_default_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_2: "f32[256]" = torch.ops.prims.convert_element_type.default(iota_default_1, torch.float32);  iota_default_1 = None
        mul_tensor_4: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.0);  convert_element_type_default_2 = None
        clamp_min_default_1: "f32[256]" = torch.ops.aten.clamp_min.default(mul_tensor_4, 0.0);  mul_tensor_4 = None
        convert_element_type_default_3: "i64[256]" = torch.ops.prims.convert_element_type.default(clamp_min_default_1, torch.int64)
        add_tensor_3: "i64[256]" = torch.ops.aten.add.Tensor(convert_element_type_default_3, 1)
        clamp_max_default_1: "i64[256]" = torch.ops.aten.clamp_max.default(add_tensor_3, 255);  add_tensor_3 = None
        _unsafe_index_tensor: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, clamp_max_default, clamp_max_default_1])
        _unsafe_index_tensor_1: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, clamp_max_default, convert_element_type_default_3]);  clamp_max_default = None
        sub_tensor_1: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor, _unsafe_index_tensor_1);  _unsafe_index_tensor = None
        sub_tensor_2: "f32[256]" = torch.ops.aten.sub.Tensor(clamp_min_default_1, convert_element_type_default_3);  clamp_min_default_1 = None
        clamp_min_default_2: "f32[256]" = torch.ops.aten.clamp_min.default(sub_tensor_2, 0.0);  sub_tensor_2 = None
        clamp_max_default_2: "f32[256]" = torch.ops.aten.clamp_max.default(clamp_min_default_2, 1.0);  clamp_min_default_2 = None
        mul_tensor_5: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_1, clamp_max_default_2);  sub_tensor_1 = None
        add_tensor_4: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_1, mul_tensor_5);  _unsafe_index_tensor_1 = mul_tensor_5 = None
        _unsafe_index_tensor_2: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, convert_element_type_default_1, clamp_max_default_1]);  clamp_max_default_1 = None
        _unsafe_index_tensor_3: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, convert_element_type_default_1, convert_element_type_default_3]);  relu_default = convert_element_type_default_3 = None
        sub_tensor_3: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_2, _unsafe_index_tensor_3);  _unsafe_index_tensor_2 = None
        mul_tensor_6: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_3, clamp_max_default_2);  sub_tensor_3 = clamp_max_default_2 = None
        add_tensor_5: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_3, mul_tensor_6);  _unsafe_index_tensor_3 = mul_tensor_6 = None
        sub_tensor_4: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(add_tensor_4, add_tensor_5);  add_tensor_4 = None
        sub_tensor_5: "f32[256, 1]" = torch.ops.aten.sub.Tensor(view_default, convert_element_type_default_1);  view_default = convert_element_type_default_1 = None
        clamp_min_default_3: "f32[256, 1]" = torch.ops.aten.clamp_min.default(sub_tensor_5, 0.0);  sub_tensor_5 = None
        clamp_max_default_3: "f32[256, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default_3, 1.0);  clamp_min_default_3 = None
        mul_tensor_7: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_4, clamp_max_default_3);  sub_tensor_4 = clamp_max_default_3 = None
        add_tensor_6: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_7);  add_tensor_5 = mul_tensor_7 = None
        unsqueeze_default_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_default_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_6: "f32[1, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_default_9);  convolution_58 = unsqueeze_default_9 = None
        add_tensor_7: "f32[64]" = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_default_1: "f32[64]" = torch.ops.aten.sqrt.default(add_tensor_7);  add_tensor_7 = None
        reciprocal_default_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, -1);  mul_tensor_8 = None
        unsqueeze_default_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_9: "f32[1, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_11);  sub_tensor_6 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_default_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_10: "f32[1, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, unsqueeze_default_13);  mul_tensor_9 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_default_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_8: "f32[1, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor_10, unsqueeze_default_15);  mul_tensor_10 = unsqueeze_default_15 = None
        relu_default_1: "f32[1, 64, 128, 128]" = torch.ops.aten.relu.default(add_tensor_8);  add_tensor_8 = None
        iota_default_2: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_4: "f32[256]" = torch.ops.prims.convert_element_type.default(iota_default_2, torch.float32);  iota_default_2 = None
        mul_tensor_11: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.4980392156862745);  convert_element_type_default_4 = None
        clamp_min_default_4: "f32[256]" = torch.ops.aten.clamp_min.default(mul_tensor_11, 0.0);  mul_tensor_11 = None
        view_default_1: "f32[256, 1]" = torch.ops.aten.view.default(clamp_min_default_4, _shape_param_1);  clamp_min_default_4 = _shape_param_1 = None
        convert_element_type_default_5: "i64[256, 1]" = torch.ops.prims.convert_element_type.default(view_default_1, torch.int64)
        add_tensor_9: "i64[256, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default_5, 1)
        clamp_max_default_4: "i64[256, 1]" = torch.ops.aten.clamp_max.default(add_tensor_9, 127);  add_tensor_9 = None
        iota_default_3: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_6: "f32[256]" = torch.ops.prims.convert_element_type.default(iota_default_3, torch.float32);  iota_default_3 = None
        mul_tensor_12: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 0.4980392156862745);  convert_element_type_default_6 = None
        clamp_min_default_5: "f32[256]" = torch.ops.aten.clamp_min.default(mul_tensor_12, 0.0);  mul_tensor_12 = None
        convert_element_type_default_7: "i64[256]" = torch.ops.prims.convert_element_type.default(clamp_min_default_5, torch.int64)
        add_tensor_10: "i64[256]" = torch.ops.aten.add.Tensor(convert_element_type_default_7, 1)
        clamp_max_default_5: "i64[256]" = torch.ops.aten.clamp_max.default(add_tensor_10, 127);  add_tensor_10 = None
        _unsafe_index_tensor_4: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_1, [None, None, clamp_max_default_4, clamp_max_default_5])
        _unsafe_index_tensor_5: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_1, [None, None, clamp_max_default_4, convert_element_type_default_7]);  clamp_max_default_4 = None
        sub_tensor_7: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_4, _unsafe_index_tensor_5);  _unsafe_index_tensor_4 = None
        sub_tensor_8: "f32[256]" = torch.ops.aten.sub.Tensor(clamp_min_default_5, convert_element_type_default_7);  clamp_min_default_5 = None
        clamp_min_default_6: "f32[256]" = torch.ops.aten.clamp_min.default(sub_tensor_8, 0.0);  sub_tensor_8 = None
        clamp_max_default_6: "f32[256]" = torch.ops.aten.clamp_max.default(clamp_min_default_6, 1.0);  clamp_min_default_6 = None
        mul_tensor_13: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_7, clamp_max_default_6);  sub_tensor_7 = None
        add_tensor_11: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_5, mul_tensor_13);  _unsafe_index_tensor_5 = mul_tensor_13 = None
        _unsafe_index_tensor_6: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_1, [None, None, convert_element_type_default_5, clamp_max_default_5]);  clamp_max_default_5 = None
        _unsafe_index_tensor_7: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_1, [None, None, convert_element_type_default_5, convert_element_type_default_7]);  relu_default_1 = convert_element_type_default_7 = None
        sub_tensor_9: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_6, _unsafe_index_tensor_7);  _unsafe_index_tensor_6 = None
        mul_tensor_14: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_9, clamp_max_default_6);  sub_tensor_9 = clamp_max_default_6 = None
        add_tensor_12: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_7, mul_tensor_14);  _unsafe_index_tensor_7 = mul_tensor_14 = None
        sub_tensor_10: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(add_tensor_11, add_tensor_12);  add_tensor_11 = None
        sub_tensor_11: "f32[256, 1]" = torch.ops.aten.sub.Tensor(view_default_1, convert_element_type_default_5);  view_default_1 = convert_element_type_default_5 = None
        clamp_min_default_7: "f32[256, 1]" = torch.ops.aten.clamp_min.default(sub_tensor_11, 0.0);  sub_tensor_11 = None
        clamp_max_default_7: "f32[256, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default_7, 1.0);  clamp_min_default_7 = None
        mul_tensor_15: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_10, clamp_max_default_7);  sub_tensor_10 = clamp_max_default_7 = None
        add_tensor_13: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(add_tensor_12, mul_tensor_15);  add_tensor_12 = mul_tensor_15 = None
        unsqueeze_default_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_default_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        sub_tensor_12: "f32[1, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_default_17);  convolution_59 = unsqueeze_default_17 = None
        add_tensor_14: "f32[64]" = torch.ops.aten.add.Tensor(arg298_1, 1e-05);  arg298_1 = None
        sqrt_default_2: "f32[64]" = torch.ops.aten.sqrt.default(add_tensor_14);  add_tensor_14 = None
        reciprocal_default_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_default_2);  sqrt_default_2 = None
        mul_tensor_16: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 1);  reciprocal_default_2 = None
        unsqueeze_default_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, -1);  mul_tensor_16 = None
        unsqueeze_default_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        mul_tensor_17: "f32[1, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_tensor_12, unsqueeze_default_19);  sub_tensor_12 = unsqueeze_default_19 = None
        unsqueeze_default_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_default_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_18: "f32[1, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_tensor_17, unsqueeze_default_21);  mul_tensor_17 = unsqueeze_default_21 = None
        unsqueeze_default_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_default_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_15: "f32[1, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_tensor_18, unsqueeze_default_23);  mul_tensor_18 = unsqueeze_default_23 = None
        relu_default_2: "f32[1, 64, 64, 64]" = torch.ops.aten.relu.default(add_tensor_15);  add_tensor_15 = None
        iota_default_4: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_8: "f32[256]" = torch.ops.prims.convert_element_type.default(iota_default_4, torch.float32);  iota_default_4 = None
        mul_tensor_19: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 0.24705882352941178);  convert_element_type_default_8 = None
        clamp_min_default_8: "f32[256]" = torch.ops.aten.clamp_min.default(mul_tensor_19, 0.0);  mul_tensor_19 = None
        view_default_2: "f32[256, 1]" = torch.ops.aten.view.default(clamp_min_default_8, _shape_param_2);  clamp_min_default_8 = _shape_param_2 = None
        convert_element_type_default_9: "i64[256, 1]" = torch.ops.prims.convert_element_type.default(view_default_2, torch.int64)
        add_tensor_16: "i64[256, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default_9, 1)
        clamp_max_default_8: "i64[256, 1]" = torch.ops.aten.clamp_max.default(add_tensor_16, 63);  add_tensor_16 = None
        iota_default_5: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_10: "f32[256]" = torch.ops.prims.convert_element_type.default(iota_default_5, torch.float32);  iota_default_5 = None
        mul_tensor_20: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 0.24705882352941178);  convert_element_type_default_10 = None
        clamp_min_default_9: "f32[256]" = torch.ops.aten.clamp_min.default(mul_tensor_20, 0.0);  mul_tensor_20 = None
        convert_element_type_default_11: "i64[256]" = torch.ops.prims.convert_element_type.default(clamp_min_default_9, torch.int64)
        add_tensor_17: "i64[256]" = torch.ops.aten.add.Tensor(convert_element_type_default_11, 1)
        clamp_max_default_9: "i64[256]" = torch.ops.aten.clamp_max.default(add_tensor_17, 63);  add_tensor_17 = None
        _unsafe_index_tensor_8: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_2, [None, None, clamp_max_default_8, clamp_max_default_9])
        _unsafe_index_tensor_9: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_2, [None, None, clamp_max_default_8, convert_element_type_default_11]);  clamp_max_default_8 = None
        sub_tensor_13: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_8, _unsafe_index_tensor_9);  _unsafe_index_tensor_8 = None
        sub_tensor_14: "f32[256]" = torch.ops.aten.sub.Tensor(clamp_min_default_9, convert_element_type_default_11);  clamp_min_default_9 = None
        clamp_min_default_10: "f32[256]" = torch.ops.aten.clamp_min.default(sub_tensor_14, 0.0);  sub_tensor_14 = None
        clamp_max_default_10: "f32[256]" = torch.ops.aten.clamp_max.default(clamp_min_default_10, 1.0);  clamp_min_default_10 = None
        mul_tensor_21: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_13, clamp_max_default_10);  sub_tensor_13 = None
        add_tensor_18: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_9, mul_tensor_21);  _unsafe_index_tensor_9 = mul_tensor_21 = None
        _unsafe_index_tensor_10: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_2, [None, None, convert_element_type_default_9, clamp_max_default_9]);  clamp_max_default_9 = None
        _unsafe_index_tensor_11: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_2, [None, None, convert_element_type_default_9, convert_element_type_default_11]);  relu_default_2 = convert_element_type_default_11 = None
        sub_tensor_15: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_10, _unsafe_index_tensor_11);  _unsafe_index_tensor_10 = None
        mul_tensor_22: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_15, clamp_max_default_10);  sub_tensor_15 = clamp_max_default_10 = None
        add_tensor_19: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_11, mul_tensor_22);  _unsafe_index_tensor_11 = mul_tensor_22 = None
        sub_tensor_16: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(add_tensor_18, add_tensor_19);  add_tensor_18 = None
        sub_tensor_17: "f32[256, 1]" = torch.ops.aten.sub.Tensor(view_default_2, convert_element_type_default_9);  view_default_2 = convert_element_type_default_9 = None
        clamp_min_default_11: "f32[256, 1]" = torch.ops.aten.clamp_min.default(sub_tensor_17, 0.0);  sub_tensor_17 = None
        clamp_max_default_11: "f32[256, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default_11, 1.0);  clamp_min_default_11 = None
        mul_tensor_23: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_16, clamp_max_default_11);  sub_tensor_16 = clamp_max_default_11 = None
        add_tensor_20: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(add_tensor_19, mul_tensor_23);  add_tensor_19 = mul_tensor_23 = None
        unsqueeze_default_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_default_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, -1);  unsqueeze_default_24 = None
        sub_tensor_18: "f32[1, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_default_25);  convolution_60 = unsqueeze_default_25 = None
        add_tensor_21: "f32[64]" = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_default_3: "f32[64]" = torch.ops.aten.sqrt.default(add_tensor_21);  add_tensor_21 = None
        reciprocal_default_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_default_3);  sqrt_default_3 = None
        mul_tensor_24: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_default_3, 1);  reciprocal_default_3 = None
        unsqueeze_default_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_24, -1);  mul_tensor_24 = None
        unsqueeze_default_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, -1);  unsqueeze_default_26 = None
        mul_tensor_25: "f32[1, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_tensor_18, unsqueeze_default_27);  sub_tensor_18 = unsqueeze_default_27 = None
        unsqueeze_default_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_default_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, -1);  unsqueeze_default_28 = None
        mul_tensor_26: "f32[1, 64, 32, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_25, unsqueeze_default_29);  mul_tensor_25 = unsqueeze_default_29 = None
        unsqueeze_default_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_default_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, -1);  unsqueeze_default_30 = None
        add_tensor_22: "f32[1, 64, 32, 32]" = torch.ops.aten.add.Tensor(mul_tensor_26, unsqueeze_default_31);  mul_tensor_26 = unsqueeze_default_31 = None
        relu_default_3: "f32[1, 64, 32, 32]" = torch.ops.aten.relu.default(add_tensor_22);  add_tensor_22 = None
        iota_default_6: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_12: "f32[256]" = torch.ops.prims.convert_element_type.default(iota_default_6, torch.float32);  iota_default_6 = None
        mul_tensor_27: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type_default_12, 0.12156862745098039);  convert_element_type_default_12 = None
        clamp_min_default_12: "f32[256]" = torch.ops.aten.clamp_min.default(mul_tensor_27, 0.0);  mul_tensor_27 = None
        view_default_3: "f32[256, 1]" = torch.ops.aten.view.default(clamp_min_default_12, _shape_param_3);  clamp_min_default_12 = _shape_param_3 = None
        convert_element_type_default_13: "i64[256, 1]" = torch.ops.prims.convert_element_type.default(view_default_3, torch.int64)
        add_tensor_23: "i64[256, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default_13, 1)
        clamp_max_default_12: "i64[256, 1]" = torch.ops.aten.clamp_max.default(add_tensor_23, 31);  add_tensor_23 = None
        iota_default_7: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_14: "f32[256]" = torch.ops.prims.convert_element_type.default(iota_default_7, torch.float32);  iota_default_7 = None
        mul_tensor_28: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type_default_14, 0.12156862745098039);  convert_element_type_default_14 = None
        clamp_min_default_13: "f32[256]" = torch.ops.aten.clamp_min.default(mul_tensor_28, 0.0);  mul_tensor_28 = None
        convert_element_type_default_15: "i64[256]" = torch.ops.prims.convert_element_type.default(clamp_min_default_13, torch.int64)
        add_tensor_24: "i64[256]" = torch.ops.aten.add.Tensor(convert_element_type_default_15, 1)
        clamp_max_default_13: "i64[256]" = torch.ops.aten.clamp_max.default(add_tensor_24, 31);  add_tensor_24 = None
        _unsafe_index_tensor_12: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_3, [None, None, clamp_max_default_12, clamp_max_default_13])
        _unsafe_index_tensor_13: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_3, [None, None, clamp_max_default_12, convert_element_type_default_15]);  clamp_max_default_12 = None
        sub_tensor_19: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_12, _unsafe_index_tensor_13);  _unsafe_index_tensor_12 = None
        sub_tensor_20: "f32[256]" = torch.ops.aten.sub.Tensor(clamp_min_default_13, convert_element_type_default_15);  clamp_min_default_13 = None
        clamp_min_default_14: "f32[256]" = torch.ops.aten.clamp_min.default(sub_tensor_20, 0.0);  sub_tensor_20 = None
        clamp_max_default_14: "f32[256]" = torch.ops.aten.clamp_max.default(clamp_min_default_14, 1.0);  clamp_min_default_14 = None
        mul_tensor_29: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_19, clamp_max_default_14);  sub_tensor_19 = None
        add_tensor_25: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_13, mul_tensor_29);  _unsafe_index_tensor_13 = mul_tensor_29 = None
        _unsafe_index_tensor_14: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_3, [None, None, convert_element_type_default_13, clamp_max_default_13]);  clamp_max_default_13 = None
        _unsafe_index_tensor_15: "f32[1, 64, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default_3, [None, None, convert_element_type_default_13, convert_element_type_default_15]);  relu_default_3 = convert_element_type_default_15 = None
        sub_tensor_21: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_14, _unsafe_index_tensor_15);  _unsafe_index_tensor_14 = None
        mul_tensor_30: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_21, clamp_max_default_14);  sub_tensor_21 = clamp_max_default_14 = None
        add_tensor_26: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_15, mul_tensor_30);  _unsafe_index_tensor_15 = mul_tensor_30 = None
        sub_tensor_22: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(add_tensor_25, add_tensor_26);  add_tensor_25 = None
        sub_tensor_23: "f32[256, 1]" = torch.ops.aten.sub.Tensor(view_default_3, convert_element_type_default_13);  view_default_3 = convert_element_type_default_13 = None
        clamp_min_default_15: "f32[256, 1]" = torch.ops.aten.clamp_min.default(sub_tensor_23, 0.0);  sub_tensor_23 = None
        clamp_max_default_15: "f32[256, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default_15, 1.0);  clamp_min_default_15 = None
        mul_tensor_31: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_22, clamp_max_default_15);  sub_tensor_22 = clamp_max_default_15 = None
        add_tensor_27: "f32[1, 64, 256, 256]" = torch.ops.aten.add.Tensor(add_tensor_26, mul_tensor_31);  add_tensor_26 = mul_tensor_31 = None
        cat_default: "f32[1, 256, 256, 256]" = torch.ops.aten.cat.default([add_tensor_6, add_tensor_13, add_tensor_20, add_tensor_27], 1);  add_tensor_6 = add_tensor_13 = add_tensor_20 = add_tensor_27 = None
        return cat_default



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
