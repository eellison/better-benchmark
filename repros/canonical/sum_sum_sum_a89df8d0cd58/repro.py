"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_711: "f32[32768, 128]", mul_699: "f32[256, 128, 128]", addmm_1: "f32[32768, 128]", primals_12: "f32[128]", primals_13: "f32[128]", addmm_6: "f32[32768, 128]", primals_26: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:359 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(mm_711, _shape_param_0);  mm_711 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_699, reshape_default);  mul_699 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: return input_tensor * self.weight + self.bias
        sum_dim_int_list: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], True)
        reshape_default_1: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:409 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_2: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default_2, primals_12)
        add_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, primals_13);  mul_tensor = primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_3: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_6, _shape_param_3);  addmm_6 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_tensor_2: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_3, add_tensor_1);  reshape_default_3 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, add_tensor_2);  add_tensor_2 = None
        mul_tensor_2: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, primals_26);  add_tensor = primals_26 = None
        sum_dim_int_list_1: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default_4: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_4);  sum_dim_int_list_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:292 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_5: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_5);  _shape_param_5 = None
        permute_default: "f32[128, 32768]" = torch.ops.aten.permute.default(reshape_default_5, [1, 0])
        sum_dim_int_list_2: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default_5, [0], True);  reshape_default_5 = None
        reshape_default_6: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_6);  sum_dim_int_list_2 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: return input_tensor * self.weight + self.bias
        sum_dim_int_list_3: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True)
        reshape_default_7: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_7);  sum_dim_int_list_3 = _shape_param_7 = None
        mul_tensor_3: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, reshape_default_2);  reshape_default_2 = None
        mul_tensor_4: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_12);  mul_tensor_2 = primals_12 = None
        sum_dim_int_list_4: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_8: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_8);  sum_dim_int_list_4 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:409 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_9: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_9);  mul_tensor_4 = _shape_param_9 = None
        permute_default_1: "f32[128, 32768]" = torch.ops.aten.permute.default(reshape_default_9, [1, 0])
        sum_dim_int_list_5: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default_9, [0], True);  reshape_default_9 = None
        reshape_default_10: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_10);  sum_dim_int_list_5 = _shape_param_10 = None
        return (reshape_default_1, reshape_default_4, permute_default, reshape_default_6, reshape_default_7, reshape_default_8, permute_default_1, reshape_default_10)


def _default_make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    [256, 128, 128],  # _shape_param_0
    [128],  # _shape_param_1
    [256, 128, 128],  # _shape_param_2
    [256, 128, 128],  # _shape_param_3
    [128],  # _shape_param_4
    [32768, 128],  # _shape_param_5
    [128],  # _shape_param_6
    [128],  # _shape_param_7
    [128],  # _shape_param_8
    [32768, 128],  # _shape_param_9
    [128],  # _shape_param_10
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
