"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_infer
Pattern hash: 5dec7369b22e
Shape hash: 6be355f0
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
_shapes_config = "(T([64, 512, 1, 1], f32), T([512], f32), T([512], f32), T([64, 512, 1, 1], f32), T([1000, 512], f32), S([64, 32, 16, 1]), S([64, 512, 1, 1]), S([64, 512]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_19: "f32[64, 512, 1, 1]", arg59_1: "f32[512]", arg60_1: "f32[512]", relu_14: "f32[64, 512, 1, 1]", arg61_1: "f32[1000, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        reshape_default: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_19, _shape_param_0);  convolution_19 = _shape_param_0 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[64, 32, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 32, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 32, 16, 1]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        reshape_default_1: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        unsqueeze_default: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(arg59_1, 0);  arg59_1 = None
        unsqueeze_default_1: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_1: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_1, unsqueeze_default_2);  reshape_default_1 = unsqueeze_default_2 = None
        unsqueeze_default_3: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(arg60_1, 0);  arg60_1 = None
        unsqueeze_default_4: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        add_tensor_1: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_tensor_2: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(add_tensor_1, relu_14);  add_tensor_1 = relu_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_default: "f32[64, 512, 1, 1]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean_dim: "f32[64, 512, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        reshape_default_2: "f32[64, 512]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_2);  mean_dim = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute_default: "f32[512, 1000]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        return (reshape_default_2, permute_default)



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
