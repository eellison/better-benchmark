"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch14_dinov2.lvd142m_training
Pattern hash: 72aebf43c278
Shape hash: 5f09d0ce
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[32, 768]", primals_174: "f32[768]", mul_108: "f32[32, 1370, 768]", div: "f32[32, 1370, 1]", primals_173: "f32[768]", _shape_param_0, primals_171: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in global_pool_nlc, code: x = x[:, 0]  # class token
        full_default: "f32[32, 1370, 768]" = torch.ops.aten.full.default([32, 1370, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[32, 1370, 768]" = torch.ops.aten.select_scatter.default(full_default, tangents_1, 1, 0);  full_default = tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(select_scatter_default, primals_174);  select_scatter_default = primals_174 = None
        mul_tensor_1: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_108);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_108, sum_dim_int_list_1);  mul_108 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 1370, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(div, sub_tensor_1);  div = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_5: "f32[32, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, primals_173);  mul_tensor_4 = primals_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[43840, 768]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_0);  mul_tensor_5 = _shape_param_0 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn(44026, dtype=torch.float32, device='cuda').as_strided([32, 1370, 1], [1376, 1, 1]),  # div
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [43840, 768],  # _shape_param_0
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
