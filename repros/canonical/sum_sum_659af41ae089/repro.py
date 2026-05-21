"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train
Pattern hash: 659af41ae089
Shape hash: 2f2590a2
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768], f32), T([128, 768], f32), T([768], f32), T([128, 198, 768], f32), T([128, 198, 1], f32), S([25344, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 768]", mm_2: "f32[128, 768]", primals_151: "f32[768]", mul_84: "f32[128, 198, 768]", div_2: "f32[128, 198, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:114 in forward_head, code: x, x_dist = x[:, 0], x[:, 1]
        full_default: "f32[128, 198, 768]" = torch.ops.aten.full.default([128, 198, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[128, 198, 768]" = torch.ops.aten.select_scatter.default(full_default, mm, 1, 1);  mm = None
        select_scatter_default_1: "f32[128, 198, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_2, 1, 0);  full_default = mm_2 = None
        add_tensor: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(add_tensor, primals_151);  add_tensor = primals_151 = None
        mul_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_84);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_dim_int_list_1);  mul_84 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_tensor_1);  div_2 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[25344, 768]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_0);  mul_tensor_4 = _shape_param_0 = None
        return reshape_default



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
