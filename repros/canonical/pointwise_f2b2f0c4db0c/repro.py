"""
Standalone repro captured via capture_hook.
Label: torchbench_resnet152_infer
Pattern hash: f2b2f0c4db0c
Shape hash: f8166fc1
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
_shapes_config = "(T([2048], f16), T([32, 2048, 7, 7], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([32, 2048, 7, 7], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg757_1: "f16[2048]", convolution_151: "f16[32, 2048, 7, 7]", arg758_1: "f16[2048]", arg759_1: "f16[2048]", arg760_1: "f16[2048]", relu_144: "f16[32, 2048, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_default: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg757_1, torch.float32);  arg757_1 = None
        unsqueeze_default: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_default_1);  convolution_151 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg758_1, torch.float32);  arg758_1 = None
        add_tensor: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[2048]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg759_1, -1);  arg759_1 = None
        unsqueeze_default_5: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg760_1, -1);  arg760_1 = None
        unsqueeze_default_7: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[32, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_tensor_2: "f16[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, relu_144);  convert_element_type_default_2 = relu_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_default: "f16[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
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
