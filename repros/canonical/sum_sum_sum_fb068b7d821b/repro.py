"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: fb068b7d821b
Shape hash: d76bd38f
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
_shapes_config = "(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([128, 197, 1], f32), T([128, 197, 768], f32), T([768], f32), T([25216, 768], f32), S([128, 197, 768]), S([128, 197, 768]), S([768]), S([25216, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_88: "f32[25216, 768]", primals_24: "f32[768]", mul_9: "f32[128, 197, 768]", div_23: "f32[128, 197, 1]", add_128: "f32[128, 197, 768]", primals_16: "f32[768]", addmm_3: "f32[25216, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        reshape_default: "f32[128, 197, 768]" = torch.ops.aten.reshape.default(mm_88, _shape_param_0);  mm_88 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_24);  primals_24 = None
        mul_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_9);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_tensor_1);  div_23 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(reshape_default, mul_9);  mul_9 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None
        add_tensor: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(add_128, mul_tensor_4);  add_128 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_6: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(add_tensor, primals_16);  primals_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[128, 197, 768]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_1);  addmm_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_7: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None
        sum_dim_int_list_4: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_3: "f32[25216, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        permute_default: "f32[768, 25216]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0])
        sum_dim_int_list_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default_3, [0], True);  reshape_default_3 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_4);  sum_dim_int_list_5 = _shape_param_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, reshape_default_2, permute_default, reshape_default_4)



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
