"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_training
Pattern hash: 1866fa73bbe4
Shape hash: 2f00cf1d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_167: "f32[32, 80, 56, 56]", primals_8: "f32[80]", convolution_1: "f32[32, 80, 56, 56]", getitem_3: "f32[32, 56, 56, 1]", rsqrt_1: "f32[32, 56, 56, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[32, 56, 56, 80]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 3, 1]);  getitem_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[32, 56, 56, 80]" = torch.ops.aten.mul.Tensor(permute_default, primals_8);  permute_default = primals_8 = None
        mul_tensor_1: "f32[32, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_tensor, 80)
        sum_dim_int_list: "f32[32, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_1: "f32[32, 56, 56, 80]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[32, 56, 56, 80]" = torch.ops.aten.sub.Tensor(permute_default_1, getitem_3);  permute_default_1 = getitem_3 = None
        mul_tensor_2: "f32[32, 56, 56, 80]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = None
        mul_tensor_3: "f32[32, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[32, 56, 56, 80]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 56, 56, 80]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[32, 56, 56, 80]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[32, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 80);  rsqrt_1 = None
        mul_tensor_5: "f32[32, 56, 56, 80]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_2: "f32[32, 80, 56, 56]" = torch.ops.aten.permute.default(mul_tensor_5, [0, 3, 1, 2]);  mul_tensor_5 = None
        return permute_default_2


def _default_make_inputs():
    return [
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([32, 80, 56, 56], [250880, 1, 4480, 80]),  # getitem_167
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([32, 80, 56, 56], [250880, 1, 4480, 80]),  # convolution_1
    torch.randn(100352, dtype=torch.float32, device='cuda').as_strided([32, 56, 56, 1], [3136, 1, 56, 56]),  # getitem_3
    torch.randn(100352, dtype=torch.float32, device='cuda').as_strided([32, 56, 56, 1], [3136, 1, 56, 56]),  # rsqrt_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
