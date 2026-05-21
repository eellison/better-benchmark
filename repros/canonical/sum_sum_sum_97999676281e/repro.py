"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train
Pattern hash: 97999676281e
Shape hash: 0b33f9aa
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
_shapes_config = "(T([25344, 768], f32), T([768], f32), T([128, 198, 768], f32), T([128, 198, 1], f32), T([128, 198, 768], f32), S([128, 198, 768]), S([25344, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_94: "f32[25344, 768]", primals_13: "f32[768]", mul_2: "f32[128, 198, 768]", div_25: "f32[128, 198, 1]", add_132: "f32[128, 198, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_13);  primals_13 = None
        mul_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_tensor_1);  div_25 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(reshape_default, mul_2);  mul_2 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None
        add_tensor: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(add_132, mul_tensor_4);  add_132 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_1: "f32[25344, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[768, 25344]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [0], True);  reshape_default_1 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, reshape_default_2)



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
