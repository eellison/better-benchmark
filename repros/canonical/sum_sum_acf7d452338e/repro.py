"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: acf7d452338e
Shape hash: 52b31666
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
    def forward(self, mm_101: "f32[788, 768]", _shape_param_0, primals_18: "f32[768]", mul_317: "f32[4, 197, 768]", div_22: "f32[4, 197, 1]", add_161: "f32[4, 197, 768]", _shape_param_1, primals_16: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        reshape_default: "f32[197, 4, 768]" = torch.ops.aten.reshape.default(mm_101, _shape_param_0);  mm_101 = _shape_param_0 = None
        permute_default: "f32[4, 197, 768]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        mul_tensor: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(permute_default, primals_18);  permute_default = primals_18 = None
        mul_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_317);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_317, sum_dim_int_list_1);  mul_317 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_tensor_1);  div_22 = sub_tensor_1 = None
        add_tensor: "f32[4, 197, 768]" = torch.ops.aten.add.Tensor(add_161, mul_tensor_4);  add_161 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        reshape_default_1: "f32[788, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default_1: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_default_2: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [197, 4, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_317
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    [788, 768],  # _shape_param_1
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
