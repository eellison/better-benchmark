"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: dc8503d4a5a5
Shape hash: 93ffdd25
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
_shapes_config = "(T([2048], f32), T([32, 2048, 7, 7], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([32, 2048, 7, 7], f32), T([128, 2048], f32), S([32, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, arg23_1: "f32[2048]", convolution_4: "f32[32, 2048, 7, 7]", arg24_1: "f32[2048]", arg25_1: "f32[2048]", arg26_1: "f32[2048]", relu_1: "f32[32, 2048, 7, 7]", arg27_1: "f32[128, 2048]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_default: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_default_1: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_default_1);  convolution_4 = unsqueeze_default_1 = None
        add_tensor: "f32[2048]" = torch.ops.aten.add.Tensor(arg24_1, 1e-05);  arg24_1 = None
        sqrt_default: "f32[2048]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_default_5: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_default_7: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_tensor_2: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, relu_1);  add_tensor_1 = relu_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_default: "f32[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean_dim: "f32[32, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        reshape_default: "f32[32, 2048]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute_default: "f32[2048, 128]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        return (reshape_default, permute_default)



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
