"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_inference
Pattern hash: 51329d516138
Shape hash: 89fe0435
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_657: "f32[32, 7, 7, 1024]", getitem_177: "f32[32, 7, 7, 1]", getitem_176: "f32[32, 7, 7, 1]", arg361_1: "f32[1024]", arg362_1: "f32[1024]", arg363_1: "f32[1000, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        sub_tensor: "f32[32, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_657, getitem_177);  view_657 = getitem_177 = None
        add_tensor: "f32[32, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_default: "f32[32, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[32, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg361_1);  mul_tensor = arg361_1 = None
        add_tensor_1: "f32[32, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg362_1);  mul_tensor_1 = arg362_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:65 in forward, code: return x.mean(self.dim, keepdim=not self.flatten)
        mean_dim: "f32[32, 1024]" = torch.ops.aten.mean.dim(add_tensor_1, [1, 2]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        return (mean_dim, permute_default)


def _default_make_inputs():
    return [
    torch.randn([32, 7, 7, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([32, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
