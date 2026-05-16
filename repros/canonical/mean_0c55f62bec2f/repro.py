"""
Standalone repro captured via capture_hook.
Label: timm_resnet50
Pattern hash: 0c55f62bec2f
Shape hash: 9e368c21
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg262_1: "f32[2048]", convolution_52: "f32[8, 2048, 7, 7]", arg263_1: "f32[2048]", arg264_1: "f32[2048]", arg265_1: "f32[2048]", relu_45: "f32[8, 2048, 7, 7]", _shape_param_0, arg266_1: "f32[1000, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:258 in forward, code: x = self.bn3(x)
        unsqueeze_default: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_default_1: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_default_1);  convolution_52 = unsqueeze_default_1 = None
        add_tensor: "f32[2048]" = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_default: "f32[2048]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_default_5: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_default_7: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:268 in forward, code: x += shortcut
        add_tensor_2: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, relu_45);  add_tensor_1 = relu_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:269 in forward, code: x = self.act3(x)
        relu_default: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        as_strided_default: "f32[8, 2048, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [8, 2048, 1, 1], [2048, 1, 2048, 2048]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[8, 2048]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnet.py:772 in forward_head, code: return x if pre_logits else self.fc(x)
        permute_default: "f32[2048, 1000]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([8, 2048, 7, 7], [100352, 1, 14336, 2048]),  # convolution_52
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([8, 2048, 7, 7], [100352, 1, 14336, 2048]),  # relu_45
    [8, 2048],  # _shape_param_0
    torch.randn([1000, 2048], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
