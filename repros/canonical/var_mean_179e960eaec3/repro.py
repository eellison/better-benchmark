"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: 179e960eaec3
Shape hash: d5b7b605
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
_shapes_config = "(T([8, 64, 320, 479], f32), T([64], f32), T([64], f32), T([8, 64, 640, 959], f32), S([640, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_15: "f32[8, 64, 320, 479]", primals_112: "f32[64]", primals_113: "f32[64]", relu_1: "f32[8, 64, 640, 959]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 64, 320, 479]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_1);  convolution_15 = getitem_1 = None
        mul_tensor: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[8, 64, 320, 479]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[8, 64, 320, 479]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[8, 64, 320, 479]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:58 in forward, code: x1 = self.up(x1)
        iota_default: "i64[640]" = torch.ops.prims.iota.default(640, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default: "f32[640]" = torch.ops.prims.convert_element_type.default(iota_default, torch.float32);  iota_default = None
        mul_tensor_2: "f32[640]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.49921752738654146);  convert_element_type_default = None
        clamp_min_default: "f32[640]" = torch.ops.aten.clamp_min.default(mul_tensor_2, 0.0);  mul_tensor_2 = None
        reshape_default: "f32[640, 1]" = torch.ops.aten.reshape.default(clamp_min_default, _shape_param_0);  clamp_min_default = _shape_param_0 = None
        convert_element_type_default_1: "i64[640, 1]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.int64)
        add_tensor_2: "i64[640, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1)
        clamp_max_default: "i64[640, 1]" = torch.ops.aten.clamp_max.default(add_tensor_2, 319);  add_tensor_2 = None
        iota_default_1: "i64[958]" = torch.ops.prims.iota.default(958, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_2: "f32[958]" = torch.ops.prims.convert_element_type.default(iota_default_1, torch.float32);  iota_default_1 = None
        mul_tensor_3: "f32[958]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.4994775339602926);  convert_element_type_default_2 = None
        clamp_min_default_1: "f32[958]" = torch.ops.aten.clamp_min.default(mul_tensor_3, 0.0);  mul_tensor_3 = None
        convert_element_type_default_3: "i64[958]" = torch.ops.prims.convert_element_type.default(clamp_min_default_1, torch.int64)
        add_tensor_3: "i64[958]" = torch.ops.aten.add.Tensor(convert_element_type_default_3, 1)
        clamp_max_default_1: "i64[958]" = torch.ops.aten.clamp_max.default(add_tensor_3, 478);  add_tensor_3 = None
        _unsafe_index_tensor: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, convert_element_type_default_1, convert_element_type_default_3])
        _unsafe_index_tensor_1: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, convert_element_type_default_1, clamp_max_default_1])
        _unsafe_index_tensor_2: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, clamp_max_default, convert_element_type_default_3])
        _unsafe_index_tensor_3: "f32[8, 64, 640, 958]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, clamp_max_default, clamp_max_default_1]);  relu_default = clamp_max_default = clamp_max_default_1 = None
        sub_tensor_1: "f32[958]" = torch.ops.aten.sub.Tensor(clamp_min_default_1, convert_element_type_default_3);  clamp_min_default_1 = convert_element_type_default_3 = None
        clamp_min_default_2: "f32[958]" = torch.ops.aten.clamp_min.default(sub_tensor_1, 0.0);  sub_tensor_1 = None
        clamp_max_default_2: "f32[958]" = torch.ops.aten.clamp_max.default(clamp_min_default_2, 1.0);  clamp_min_default_2 = None
        sub_tensor_2: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_1, _unsafe_index_tensor);  _unsafe_index_tensor_1 = None
        mul_tensor_4: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_tensor_2, clamp_max_default_2);  sub_tensor_2 = None
        add_tensor_4: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor, mul_tensor_4);  _unsafe_index_tensor = mul_tensor_4 = None
        sub_tensor_3: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(_unsafe_index_tensor_3, _unsafe_index_tensor_2);  _unsafe_index_tensor_3 = None
        mul_tensor_5: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_tensor_3, clamp_max_default_2);  sub_tensor_3 = clamp_max_default_2 = None
        add_tensor_5: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(_unsafe_index_tensor_2, mul_tensor_5);  _unsafe_index_tensor_2 = mul_tensor_5 = None
        sub_tensor_4: "f32[640, 1]" = torch.ops.aten.sub.Tensor(reshape_default, convert_element_type_default_1);  reshape_default = convert_element_type_default_1 = None
        clamp_min_default_3: "f32[640, 1]" = torch.ops.aten.clamp_min.default(sub_tensor_4, 0.0);  sub_tensor_4 = None
        clamp_max_default_3: "f32[640, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default_3, 1.0);  clamp_min_default_3 = None
        sub_tensor_5: "f32[8, 64, 640, 958]" = torch.ops.aten.sub.Tensor(add_tensor_5, add_tensor_4);  add_tensor_5 = None
        mul_tensor_6: "f32[8, 64, 640, 958]" = torch.ops.aten.mul.Tensor(sub_tensor_5, clamp_max_default_3);  sub_tensor_5 = clamp_max_default_3 = None
        add_tensor_6: "f32[8, 64, 640, 958]" = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_6);  add_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[8, 64, 640, 959]" = torch.ops.aten.constant_pad_nd.default(add_tensor_6, [0, 1, 0, 0], 0.0);  add_tensor_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        cat_default: "f32[8, 128, 640, 959]" = torch.ops.aten.cat.default([relu_1, constant_pad_nd_default], 1);  relu_1 = constant_pad_nd_default = None
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
