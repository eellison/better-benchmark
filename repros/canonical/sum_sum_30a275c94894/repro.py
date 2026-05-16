"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_training
Pattern hash: 30a275c94894
Shape hash: 85f15f9d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_50: "f32[4096, 768]", _shape_param_0, mul_428: "f32[8, 512, 768]", primals_11: "f32[768]", mul_4: "f32[8, 512, 768]", div_26: "f32[8, 512, 1]", full_default_5: "f32[8, 512, 768, 2]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_50, _shape_param_0);  mm_50 = _shape_param_0 = None
        add_tensor: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_428, reshape_default);  mul_428 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        mul_tensor: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, primals_11);  add_tensor = primals_11 = None
        mul_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_dim_int_list_1);  mul_4 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_tensor_1);  div_26 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_scatter_default: "f32[8, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_tensor_4, 3, 0);  full_default_5 = mul_tensor_4 = None
        return select_scatter_default


def _default_make_inputs():
    return [
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [8, 512, 768],  # _shape_param_0
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768, 2], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
