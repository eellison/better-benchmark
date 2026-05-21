"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train
Pattern hash: 7c49c83ea03e
Shape hash: 152ed363
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
_shapes_config = "(T([512, 232, 7, 7], f32), T([232], f32), T([232], f32), T([512, 232, 7, 7], f32), T([232], f32), T([232], f32), S([512, 2, 232, 7, 7]), S([512, 464, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_42: "f32[512, 232, 7, 7]", primals_258: "f32[232]", primals_259: "f32[232]", convolution_45: "f32[512, 232, 7, 7]", primals_276: "f32[232]", primals_277: "f32[232]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 232, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 232, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 232, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 232, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_1);  convolution_42 = getitem_1 = None
        mul_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1);  primals_258 = None
        unsqueeze_default_1: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_default_3: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 232, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 232, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 232, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 232, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_3);  convolution_45 = getitem_3 = None
        mul_tensor_2: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
        unsqueeze_default_5: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_default_7: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        relu_default_1: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        cat_default: "f32[512, 464, 7, 7]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        reshape_default: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_default: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3, 4]);  reshape_default = None
        clone_default: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        reshape_default_1: "f32[512, 464, 7, 7]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default_1, 232, 1);  reshape_default_1 = None
        getitem_4: "f32[512, 232, 7, 7]" = split_tensor[0]
        getitem_5: "f32[512, 232, 7, 7]" = split_tensor[1];  split_tensor = None
        return (getitem_4, getitem_5)



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
