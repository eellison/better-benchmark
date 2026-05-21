"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train
Pattern hash: 366a09721107
Shape hash: aecf7099
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
    def forward(self, avg_pool2d_2: "f32[64, 512, 7, 7]", convolution_89: "f32[64, 32, 7, 7]", convolution_91: "f32[64, 32, 7, 7]", convolution_93: "f32[64, 32, 7, 7]", convolution_95: "f32[64, 32, 7, 7]", convolution_97: "f32[64, 32, 7, 7]", primals_593: "f32[672]", primals_594: "f32[672]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_default: "f32[64, 672, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97], 1);  avg_pool2d_2 = convolution_89 = convolution_91 = convolution_93 = convolution_95 = convolution_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 672, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 672, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[64, 672, 7, 7]" = torch.ops.aten.sub.Tensor(cat_default, getitem_1);  cat_default = getitem_1 = None
        mul_tensor: "f32[64, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_593, -1);  primals_593 = None
        unsqueeze_default_1: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[64, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_594, -1);  primals_594 = None
        unsqueeze_default_3: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[64, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[64, 672, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return relu_default


def _default_make_inputs():
    return [
    torch.randn([64, 512, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([64, 32, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([64, 32, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([64, 32, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([64, 32, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([64, 32, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
