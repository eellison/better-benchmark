"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: a7a6ca40df0e
Shape hash: 313443b3
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_128: "f32[1, 7, 7, 768]", getitem_57: "f32[1, 7, 7, 1]", getitem_56: "f32[1, 7, 7, 1]", arg182_1: "f32[768]", arg183_1: "f32[768]", _shape_param_0, arg184_1: "f32[1000, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:609 in forward, code: x = self.norm(x)
        sub_tensor: "f32[1, 7, 7, 768]" = torch.ops.aten.sub.Tensor(add_128, getitem_57);  add_128 = getitem_57 = None
        add_tensor: "f32[1, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_default: "f32[1, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 7, 7, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg182_1);  mul_tensor = arg182_1 = None
        add_tensor_1: "f32[1, 7, 7, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg183_1);  mul_tensor_1 = arg183_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:321 in forward, code: return torch.permute(x, self.dims)
        permute_default: "f32[1, 768, 7, 7]" = torch.ops.aten.permute.default(add_tensor_1, [0, 3, 1, 2]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:611 in forward, code: x = self.avgpool(x)
        mean_dim: "f32[1, 768, 1, 1]" = torch.ops.aten.mean.dim(permute_default, [-1, -2], True);  permute_default = None
        as_strided_default: "f32[1, 768, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [1, 768, 1, 1], [768, 1, 768, 768]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:612 in forward, code: x = self.flatten(x)
        reshape_default: "f32[1, 768]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:613 in forward, code: x = self.head(x)
        permute_default_1: "f32[768, 1000]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [1, 768],  # _shape_param_0
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
