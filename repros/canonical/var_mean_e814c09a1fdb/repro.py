"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train_000
Pattern hash: e814c09a1fdb
Shape hash: ae362444
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
_shapes_config = "(T([8, 64, 320, 479], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([8, 64, 640, 959], f32), S([640, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_15: "f32[8, 64, 320, 479]", arg109_1: "f32[64]", arg110_1: "f32[64]", arg111_1: "f32[64]", arg112_1: "f32[64]", relu_1: "f32[8, 64, 640, 959]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_1);  convolution_15 = None
        mul_tensor: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(arg109_1, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000008155017088);  squeeze_dims_1 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg110_1, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[8, 64, 320, 479]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[8, 64, 320, 479]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        iota_default: "i64[640]" = torch.ops.prims.iota.default(640, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default: "f32[640]" = torch.ops.prims.convert_element_type.default(iota_default, torch.float32);  iota_default = None
        mul_tensor_7: "f32[640]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.49921752738654146);  convert_element_type_default = None
        clamp_min_default: "f32[640]" = torch.ops.aten.clamp_min.default(mul_tensor_7, 0.0);  mul_tensor_7 = None
        view_default: "f32[640, 1]" = torch.ops.aten.view.default(clamp_min_default, _shape_param_0);  clamp_min_default = _shape_param_0 = None
        convert_element_type_default_1: "i64[640, 1]" = torch.ops.prims.convert_element_type.default(view_default, torch.int64)
        add_tensor_4: "i64[640, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1)
        clamp_max_default: "i64[640, 1]" = torch.ops.aten.clamp_max.default(add_tensor_4, 319);  add_tensor_4 = None
        iota_default_1: "i64[958]" = torch.ops.prims.iota.default(958, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_2: "f32[958]" = torch.ops.prims.convert_element_type.default(iota_default_1, torch.float32);  iota_default_1 = None
        mul_tensor_8: "f32[958]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.4994775339602926);  convert_element_type_default_2 = None
        clamp_min_default_1: "f32[958]" = torch.ops.aten.clamp_min.default(mul_tensor_8, 0.0);  mul_tensor_8 = None
        convert_element_type_default_3: "i64[958]" = torch.ops.prims.convert_element_type.default(clamp_min_default_1, torch.int64)
        add_tensor_5: "i64[958]" = torch.ops.aten.add.Tensor(convert_element_type_default_3, 1)
        clamp_max_default_1: "i64[958]" = torch.ops.aten.clamp_max.default(add_tensor_5, 478);  add_tensor_5 = None
        _unsafe_index_tensor: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, convert_element_type_default_1, convert_element_type_default_3])
        _unsafe_index_tensor_1: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, convert_element_type_default_1, clamp_max_default_1])
        _unsafe_index_tensor_2: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, clamp_max_default, convert_element_type_default_3])
        _unsafe_index_tensor_3: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, clamp_max_default, clamp_max_default_1]);  relu_default = clamp_max_default = clamp_max_default_1 = None
        sub_tensor_1: "f32[958]" = torch.ops.aten.sub.Tensor(clamp_min_default_1, convert_element_type_default_3);  clamp_min_default_1 = convert_element_type_default_3 = None
        clamp_min_default_2: "f32[958]" = torch.ops.aten.clamp_min.default(sub_tensor_1, 0.0);  sub_tensor_1 = None
        clamp_max_default_2: "f32[958]" = torch.ops.aten.clamp_max.default(clamp_min_default_2, 1.0);  clamp_min_default_2 = None
        sub_tensor_2: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_1, _unsafe_index_tensor);  _unsafe_index_tensor_1 = None
        mul_tensor_9: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_tensor_2, clamp_max_default_2);  sub_tensor_2 = None
        add_tensor_6: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor, mul_tensor_9);  _unsafe_index_tensor = mul_tensor_9 = None
        sub_tensor_3: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_3, _unsafe_index_tensor_2);  _unsafe_index_tensor_3 = None
        mul_tensor_10: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_tensor_3, clamp_max_default_2);  sub_tensor_3 = clamp_max_default_2 = None
        add_tensor_7: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_2, mul_tensor_10);  _unsafe_index_tensor_2 = mul_tensor_10 = None
        sub_tensor_4: "f32[640, 1]" = torch.ops.aten.sub.Tensor(view_default, convert_element_type_default_1);  view_default = convert_element_type_default_1 = None
        clamp_min_default_3: "f32[640, 1]" = torch.ops.aten.clamp_min.default(sub_tensor_4, 0.0);  sub_tensor_4 = None
        clamp_max_default_3: "f32[640, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default_3, 1.0);  clamp_min_default_3 = None
        sub_tensor_5: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(add_tensor_7, add_tensor_6);  add_tensor_7 = None
        mul_tensor_11: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_tensor_5, clamp_max_default_3);  sub_tensor_5 = clamp_max_default_3 = None
        add_tensor_8: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(add_tensor_6, mul_tensor_11);  add_tensor_6 = mul_tensor_11 = None
        constant_pad_nd_default: "f32[8, 64, 640, 959]" = torch.ops.aten.constant_pad_nd.default(add_tensor_8, [0, 1, 0, 0], 0.0);  add_tensor_8 = None
        cat_default: "f32[8, 128, 640, 959]" = torch.ops.aten.cat.default([relu_1, constant_pad_nd_default], 1);  relu_1 = constant_pad_nd_default = None
        copy__default: "f32[64]" = torch.ops.aten.copy_.default(arg109_1, add_tensor_1);  arg109_1 = add_tensor_1 = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(arg110_1, add_tensor_2);  arg110_1 = add_tensor_2 = None
        return (cat_default, copy__default, copy__default_1)



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
