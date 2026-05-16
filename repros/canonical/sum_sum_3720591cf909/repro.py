"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_training
Pattern hash: 3720591cf909
Shape hash: 686423db
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_92: "f32[6304, 768]", _shape_param_0, primals_17: "f32[768]", addmm_1: "f32[6304, 768]", _shape_param_1, primals_5: "f32[768]", cat: "f32[32, 197, 768]", getitem_10: "f32[32, 197, 1]", rsqrt_1: "f32[32, 197, 1]", add_129: "f32[32, 197, 768]", _shape_param_2, primals_14: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(mm_92, _shape_param_0);  mm_92 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_17);  reshape_default = primals_17 = None
        mul_tensor_1: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_1: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_2: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(primals_5, reshape_default_1);  reshape_default_1 = None
        add_tensor: "f32[32, 197, 768]" = torch.ops.aten.add.Tensor(cat, mul_tensor_2);  cat = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[32, 197, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_10);  add_tensor = getitem_10 = None
        mul_tensor_3: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = None
        mul_tensor_4: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_3);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 197, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[32, 197, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        mul_tensor_6: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[32, 197, 768]" = torch.ops.aten.add.Tensor(add_129, mul_tensor_6);  add_129 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_7: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_5);  add_tensor_1 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_2: "f32[6304, 768]" = torch.ops.aten.reshape.default(mul_tensor_7, _shape_param_2);  mul_tensor_7 = _shape_param_2 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_1
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [6304, 768],  # _shape_param_2
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
