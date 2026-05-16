"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['256', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['256', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_default: "f32[32768, 512]", primals_1116: "f32[512]", addmm_361: "f32[32768, 512]", getitem_1: "f32[256, 128, 1]", rsqrt: "f32[256, 128, 1]", full_default_3: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:632 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_default, [256, 128, 512]);  mm_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:611 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(reshape_default, primals_1116);  primals_1116 = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:609 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_361, [256, 128, 512]);  addmm_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:610 in forward, code: hidden_states = self.transform_act_fn(hidden_states)
        relu_default: "f32[256, 128, 512]" = torch.ops.aten.relu.default(reshape_default_1);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:611 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(relu_default, getitem_1);  getitem_1 = None
        mul_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 512);  rsqrt = None
        mul_tensor_5: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:610 in forward, code: hidden_states = self.transform_act_fn(hidden_states)
        le_scalar: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[256, 128, 512]" = torch.ops.aten.where.self(le_scalar, full_default_3, mul_tensor_5);  le_scalar = full_default_3 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:609 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[32768, 512]" = torch.ops.aten.reshape.default(where_self, [32768, 512]);  where_self = None
        permute_default: "f32[512, 32768]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0])
        sum_dim_int_list_4: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [0], True);  reshape_default_2 = None
        reshape_default_3: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [512]);  sum_dim_int_list_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
