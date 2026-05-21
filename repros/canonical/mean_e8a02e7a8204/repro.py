"""
Standalone repro captured via capture_hook.
Label: torchbench_mnasnet1_0_infer
Pattern hash: e8a02e7a8204
Shape hash: 5577c7ff
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
    def forward(self, arg257_1: "f16[1280]", convolution_51: "f16[256, 1280, 7, 7]", arg258_1: "f16[1280]", arg259_1: "f16[1280]", arg260_1: "f16[1280]", arg261_1: "f16[1000, 1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:159 in forward, code: x = self.layers(x)
        convert_element_type_default: "f32[1280]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.float32);  arg257_1 = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[256, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_default_1);  convolution_51 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[1280]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
        add_tensor: "f32[1280]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[1280]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[1280, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_default_5: "f16[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[1280, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_default_7: "f16[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[256, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[256, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        relu_default: "f16[256, 1280, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:161 in forward, code: x = x.mean([2, 3])
        mean_dim: "f16[256, 1280]" = torch.ops.aten.mean.dim(relu_default, [2, 3]);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:162 in forward, code: return self.classifier(x)
        permute_default: "f16[1280, 1000]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        return (mean_dim, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1280], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1280, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([1280], dtype=torch.float16, device='cuda'),
    torch.randn([1280], dtype=torch.float16, device='cuda'),
    torch.randn([1280], dtype=torch.float16, device='cuda'),
    torch.randn([1000, 1280], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
