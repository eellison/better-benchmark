"""
Standalone repro captured via capture_hook.
Label: resnet18_training
Pattern hash: 28098460082b
Shape hash: ba725f38
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_30: "f32[1, 256, 1, 1]", convolution_14: "f32[4, 256, 14, 14]", getitem_31: "f32[1, 256, 1, 1]", primals_90: "f32[256]", primals_91: "f32[256]", relu_10: "f32[4, 256, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_tensor: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_default: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31);  convolution_14 = getitem_31 = None
        mul_tensor: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
        unsqueeze_default_1: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_default_3: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_tensor_2: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, relu_10);  add_tensor_1 = relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_default: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        return relu_default


def _default_make_inputs():
    return [
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 256, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([4, 256, 14, 14], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
