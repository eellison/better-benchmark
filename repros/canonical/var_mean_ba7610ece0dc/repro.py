"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train
Pattern hash: ba7610ece0dc
Shape hash: b4e442b5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_51: "f32[512, 232, 7, 7]", primals_312: "f32[232]", primals_313: "f32[232]", getitem_122: "f32[512, 232, 7, 7]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 232, 1, 1]" = var_mean_correction[0]
        getitem_123: "f32[1, 232, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 232, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 232, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_123);  convolution_51 = getitem_123 = None
        mul_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1);  primals_312 = None
        unsqueeze_default_1: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_default_3: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        cat_default: "f32[512, 464, 7, 7]" = torch.ops.aten.cat.default([getitem_122, relu_default], 1);  getitem_122 = relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        reshape_default: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_default: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3, 4]);  reshape_default = None
        clone_default: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        reshape_default_1: "f32[512, 464, 7, 7]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default_1, 232, 1);  reshape_default_1 = None
        getitem_124: "f32[512, 232, 7, 7]" = split_tensor[0]
        getitem_125: "f32[512, 232, 7, 7]" = split_tensor[1];  split_tensor = None
        return (getitem_125, getitem_124)


def _default_make_inputs():
    return [
    torch.randn([512, 232, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn([232], dtype=torch.float32, device='cuda'),
    torch.randn(11629464, dtype=torch.float32, device='cuda').as_strided([512, 232, 7, 7], [22736, 49, 7, 1]),  # getitem_122
    [512, 2, 232, 7, 7],  # _shape_param_0
    [512, 464, 7, 7],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
