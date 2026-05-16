"""
Standalone repro captured via capture_hook.
Label: densenet121_inference
Pattern hash: 2388543e87c6
Shape hash: bac84708
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg601_1: "f32[1024]", cat_57: "f32[1, 1024, 7, 7]", arg602_1: "f32[1024]", arg603_1: "f32[1024]", arg604_1: "f32[1024]", _shape_param_0, arg605_1: "f32[1000, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg601_1, -1);  arg601_1 = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[1, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(cat_57, unsqueeze_default_1);  cat_57 = unsqueeze_default_1 = None
        add_tensor: "f32[1024]" = torch.ops.aten.add.Tensor(arg602_1, 1e-05);  arg602_1 = None
        sqrt_default: "f32[1024]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[1, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg603_1, -1);  arg603_1 = None
        unsqueeze_default_5: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[1, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg604_1, -1);  arg604_1 = None
        unsqueeze_default_7: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[1, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:214 in forward, code: out = F.relu(features, inplace=True)
        relu_default: "f32[1, 1024, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:215 in forward, code: out = F.adaptive_avg_pool2d(out, (1, 1))
        mean_dim: "f32[1, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:216 in forward, code: out = torch.flatten(out, 1)
        reshape_default: "f32[1, 1024]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:217 in forward, code: out = self.classifier(out)
        permute_default: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg605_1, [1, 0]);  arg605_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    [1, 1024],  # _shape_param_0
    torch.randn([1000, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
