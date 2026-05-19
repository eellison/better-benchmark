"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: f76b72d10699
Shape hash: 29554548
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_152: "f32[128, 80, 56, 56]", primals_26: "f32[80]", mul_14: "f32[128, 56, 56, 80]", div_90: "f32[128, 56, 56, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[128, 56, 56, 80]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 3, 1]);  getitem_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(permute_default, primals_26);  permute_default = primals_26 = None
        mul_tensor_1: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_tensor, 80)
        sum_dim_int_list: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_14);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_14, sum_dim_int_list_1);  mul_14 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 56, 56, 80]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 56, 56, 80]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 56, 56, 80]" = torch.ops.aten.mul.Tensor(div_90, sub_tensor_1);  div_90 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_1: "f32[128, 80, 56, 56]" = torch.ops.aten.permute.default(mul_tensor_4, [0, 3, 1, 2]);  mul_tensor_4 = None
        return permute_default_1


def _default_make_inputs():
    return [
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 80, 56, 56], [250880, 1, 4480, 80]),  # getitem_152
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn(32112640, dtype=torch.float32, device='cuda').as_strided([128, 56, 56, 80], [250880, 1, 4480, 56]),  # mul_14
    torch.randn(401408, dtype=torch.float32, device='cuda').as_strided([128, 56, 56, 1], [3136, 1, 56, 56]),  # div_90
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
