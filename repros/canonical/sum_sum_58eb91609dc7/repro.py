"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_training
Pattern hash: 58eb91609dc7
Shape hash: d5dab4f4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_3: "f32[32, 768]", _shape_param_0, primals_158: "f32[768]", addmm_49: "f32[32, 768]", _shape_param_1, getitem_141: "f32[32, 1, 1]", rsqrt_25: "f32[32, 1, 1]", select_scatter: "f32[32, 1, 768]", _shape_param_2, primals_156: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[32, 1, 768]" = torch.ops.aten.reshape.default(mm_3, _shape_param_0);  mm_3 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[32, 1, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_158);  reshape_default = primals_158 = None
        mul_tensor_1: "f32[32, 1, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:118 in forward, code: x = self.proj(x)
        reshape_default_1: "f32[32, 1, 768]" = torch.ops.aten.reshape.default(addmm_49, _shape_param_1);  addmm_49 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[32, 1, 768]" = torch.ops.aten.sub.Tensor(reshape_default_1, getitem_141);  reshape_default_1 = getitem_141 = None
        mul_tensor_2: "f32[32, 1, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_25);  sub_tensor = None
        mul_tensor_3: "f32[32, 1, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[32, 1, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 1, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[32, 1, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[32, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_tensor_5: "f32[32, 1, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor: "f32[32, 1, 768]" = torch.ops.aten.add.Tensor(select_scatter, mul_tensor_5);  select_scatter = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:118 in forward, code: x = self.proj(x)
        reshape_default_2: "f32[32, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    [32, 1, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    [32, 1, 768],  # _shape_param_1
    torch.randn([32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1, 768], dtype=torch.float32, device='cuda'),
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
