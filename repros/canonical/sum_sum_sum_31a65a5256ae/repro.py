"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_685: "f32[32768, 512]", mul_672: "f32[256, 128, 512]", mm_691: "f32[32768, 512]", mm_693: "f32[32768, 512]", add_16: "f32[256, 128, 512]", primals_54: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:255 in forward, code: self.value(value_tensor)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_685, _shape_param_0);  mm_685 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_672, reshape_default);  mul_672 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:409 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_1: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_691, _shape_param_1);  mm_691 = _shape_param_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None
        reshape_default_2: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_693, _shape_param_2);  mm_693 = _shape_param_2 = None
        add_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: return input_tensor * self.weight + self.bias
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True)
        reshape_default_3: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, add_16);  add_16 = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_54);  add_tensor_2 = primals_54 = None
        sum_dim_int_list_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        reshape_default_4: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_4);  sum_dim_int_list_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:372 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_5: "f32[32768, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_5);  mul_tensor_1 = _shape_param_5 = None
        permute_default: "f32[512, 32768]" = torch.ops.aten.permute.default(reshape_default_5, [1, 0])
        sum_dim_int_list_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(reshape_default_5, [0], True);  reshape_default_5 = None
        reshape_default_6: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_6);  sum_dim_int_list_2 = _shape_param_6 = None
        return (reshape_default_3, reshape_default_4, permute_default, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    [256, 128, 512],  # _shape_param_0
    [256, 128, 512],  # _shape_param_1
    [256, 128, 512],  # _shape_param_2
    [512],  # _shape_param_3
    [512],  # _shape_param_4
    [32768, 512],  # _shape_param_5
    [512],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
