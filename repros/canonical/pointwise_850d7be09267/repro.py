"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: 850d7be09267
Shape hash: ff304488
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
_shapes_config = "(T([1024], f32), T([32, 1024, 14, 14], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([32, 1024, 14, 14], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[1024]", convolution: "f32[32, 1024, 14, 14]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024]", arg6_1: "f32[32, 1024, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_default_1);  convolution = unsqueeze_default_1 = None
        add_tensor: "f32[1024]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_default: "f32[1024]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_default_5: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_default_7: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_tensor_2: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(arg6_1, add_tensor_1);  arg6_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_default: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        return relu_default



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
