"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_inference
Pattern hash: 535d83dc92de
Shape hash: bf54610b
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, clone_61: "f32[32, 256, 768]", getitem_133: "f32[32, 256, 1]", getitem_132: "f32[32, 256, 1]", arg148_1: "f32[768]", arg149_1: "f32[768]", _shape_param_0, arg153_1: "f32[1536, 768]", arg150_1: "f32[1, 1, 768]", _shape_param_1, _shape_param_2, arg151_1: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(clone_61, getitem_133);  clone_61 = getitem_133 = None
        add_tensor: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_default: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg148_1);  mul_tensor = arg148_1 = None
        add_tensor_1: "f32[32, 256, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg149_1);  mul_tensor_1 = arg149_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:104 in forward, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[768, 1536]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:101 in forward, code: q_latent = self.latent.expand(B, -1, -1)
        expand_default: "f32[32, 1, 768]" = torch.ops.aten.expand.default(arg150_1, _shape_param_1);  arg150_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_1: "f32[32, 768]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 256, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [8192, 768],  # _shape_param_0
    torch.randn([1536, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    [32, -1, -1],  # _shape_param_1
    [32, 768],  # _shape_param_2
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
