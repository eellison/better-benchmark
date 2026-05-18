"""
Standalone repro captured via capture_hook.
Label: timm_deit_tiny_patch16_224.fb_in1k_inference
Pattern hash: 947549c927a4
Shape hash: ac98fda5
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_84: "f32[32, 197, 192]", getitem_133: "f32[32, 197, 1]", getitem_132: "f32[32, 197, 1]", arg149_1: "f32[192]", arg150_1: "f32[192]", arg151_1: "f32[1000, 192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[32, 197, 192]" = torch.ops.aten.sub.Tensor(add_84, getitem_133);  add_84 = getitem_133 = None
        add_tensor: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_default: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[32, 197, 192]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 197, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, arg149_1);  mul_tensor = arg149_1 = None
        add_tensor_1: "f32[32, 197, 192]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg150_1);  mul_tensor_1 = arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in global_pool_nlc, code: x = x[:, 0]  # class token
        select_int: "f32[32, 192]" = torch.ops.aten.select.int(add_tensor_1, 1, 0);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1291 in forward_head, code: x = self.head_drop(x)
        clone_default: "f32[32, 192]" = torch.ops.aten.clone.default(select_int);  select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1292 in forward_head, code: return x if pre_logits else self.head(x)
        permute_default: "f32[192, 1000]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        return (clone_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([32, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
