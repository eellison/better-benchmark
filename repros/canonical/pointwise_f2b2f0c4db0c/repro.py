"""
Standalone repro captured via capture_hook.
Label: torchbench_resnet18_infer
Pattern hash: f2b2f0c4db0c
Shape hash: b846ec58
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
    def forward(self, arg72_1: "f16[256]", convolution_14: "f16[512, 256, 14, 14]", arg73_1: "f16[256]", arg74_1: "f16[256]", arg75_1: "f16[256]", relu_10: "f16[512, 256, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_default: "f32[256]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.float32);  arg72_1 = None
        unsqueeze_default: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_default_1);  convolution_14 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[256]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None
        add_tensor: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[256]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_default_5: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_default_7: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[512, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_tensor_2: "f16[512, 256, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, relu_10);  convert_element_type_default_2 = relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_default: "f16[512, 256, 14, 14]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        return relu_default


def _default_make_inputs():
    return [
    torch.randn([256], dtype=torch.float16, device='cuda'),
    torch.randn([512, 256, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([256], dtype=torch.float16, device='cuda'),
    torch.randn([256], dtype=torch.float16, device='cuda'),
    torch.randn([256], dtype=torch.float16, device='cuda'),
    torch.randn([512, 256, 14, 14], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
