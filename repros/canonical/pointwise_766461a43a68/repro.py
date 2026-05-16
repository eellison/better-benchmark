"""
Standalone repro captured via capture_hook.
Label: resnet18_training
Pattern hash: 766461a43a68
Shape hash: c778122e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_34: "f32[1, 512, 1, 1]", convolution_16: "f32[4, 512, 7, 7]", getitem_35: "f32[1, 512, 1, 1]", primals_102: "f32[512]", primals_103: "f32[512]", getitem_36: "f32[1, 512, 1, 1]", convolution_17: "f32[4, 512, 7, 7]", getitem_37: "f32[1, 512, 1, 1]", primals_108: "f32[512]", primals_109: "f32[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        add_tensor: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_default: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_35);  convolution_16 = getitem_35 = None
        mul_tensor: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        add_tensor_2: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_default_1: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_37);  convolution_17 = getitem_37 = None
        mul_tensor_2: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
        unsqueeze_default_5: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_default_7: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_tensor_4: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor_3);  add_tensor_1 = add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_default: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        return relu_default


def _default_make_inputs():
    return [
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
