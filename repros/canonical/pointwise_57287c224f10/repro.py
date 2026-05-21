"""
Standalone repro captured via capture_hook.
Label: torchbench_resnet18_infer
Pattern hash: 57287c224f10
Shape hash: 2b1ebf61
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
    def forward(self, arg82_1: "f16[512]", convolution_16: "f16[512, 512, 7, 7]", arg83_1: "f16[512]", arg84_1: "f16[512]", arg85_1: "f16[512]", arg87_1: "f16[512]", convolution_17: "f16[512, 512, 7, 7]", arg88_1: "f16[512]", arg89_1: "f16[512]", arg90_1: "f16[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_default: "f32[512]" = torch.ops.prims.convert_element_type.default(arg82_1, torch.float32);  arg82_1 = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_default_1);  convolution_16 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[512]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        add_tensor: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[512]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_default_5: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_default_7: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[512, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convert_element_type_default_3: "f32[512]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.float32);  arg87_1 = None
        unsqueeze_default_8: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, -1);  convert_element_type_default_3 = None
        unsqueeze_default_9: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[512, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_default_9);  convolution_17 = unsqueeze_default_9 = None
        convert_element_type_default_4: "f32[512]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.float32);  arg88_1 = None
        add_tensor_2: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 1e-05);  convert_element_type_default_4 = None
        sqrt_default_1: "f32[512]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_default_13: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_default_15: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[512, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        convert_element_type_default_5: "f16[512, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_tensor_4: "f16[512, 512, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, convert_element_type_default_5);  convert_element_type_default_2 = convert_element_type_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_default: "f16[512, 512, 7, 7]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        return relu_default


def _default_make_inputs():
    return [
    torch.randn([512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
