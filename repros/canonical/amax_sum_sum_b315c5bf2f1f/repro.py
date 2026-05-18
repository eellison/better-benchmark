"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['2048', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['2048', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '512', '2880'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=mean, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg70_1: "f32[32, 2880]", bmm_15: "f32[32, 2048, 2880]", getitem_30: "f32[2048, 4]", getitem_31: "i64[2048, 4]", add_36: "f32[4, 512, 2880]", arg71_1: "f32[2880]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:136 in forward, code: next_states = next_states + self.down_proj_bias[..., None, :]
        unsqueeze_default: "f32[32, 1, 2880]" = torch.ops.aten.unsqueeze.default(arg70_1, 1);  arg70_1 = None
        add_tensor: "f32[32, 2048, 2880]" = torch.ops.aten.add.Tensor(bmm_15, unsqueeze_default);  bmm_15 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:137 in forward, code: next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        reshape_default: "f32[32, 4, 512, 2880]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:157 in forward, code: router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        full_default: "f32[2048, 32]" = torch.ops.aten.full.default([2048, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:156 in forward, code: router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        amax_default: "f32[2048, 1]" = torch.ops.aten.amax.default(getitem_30, [1], True)
        sub_tensor: "f32[2048, 4]" = torch.ops.aten.sub.Tensor(getitem_30, amax_default);  getitem_30 = amax_default = None
        exp_default: "f32[2048, 4]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[2048, 4]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:157 in forward, code: router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        scatter_src: "f32[2048, 32]" = torch.ops.aten.scatter.src(full_default, 1, getitem_31, div_tensor);  full_default = getitem_31 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:138 in forward, code: next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        permute_default: "f32[32, 2048]" = torch.ops.aten.permute.default(scatter_src, [1, 0]);  scatter_src = None
        reshape_default_1: "f32[32, 4, 512]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        unsqueeze_default_1: "f32[32, 4, 512, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, 3);  reshape_default_1 = None
        mul_tensor: "f32[32, 4, 512, 2880]" = torch.ops.aten.mul.Tensor(reshape_default, unsqueeze_default_1);  reshape_default = unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:139 in forward, code: next_states = next_states.sum(dim=0)
        sum_dim_int_list_1: "f32[4, 512, 2880]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:387 in forward, code: hidden_states = residual + hidden_states
        add_tensor_1: "f32[4, 512, 2880]" = torch.ops.aten.add.Tensor(add_36, sum_dim_int_list_1);  add_36 = sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:59 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 2880]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_1, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:60 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_1: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(add_tensor_1, rsqrt_default);  add_tensor_1 = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:61 in forward, code: return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama
        mul_tensor_2: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(arg71_1, mul_tensor_1);  arg71_1 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:683 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default_2: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_2);  mul_tensor_2 = _shape_param_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([32, 2880], dtype=torch.float32, device='cuda'),
    torch.randn([32, 2048, 2880], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 4], dtype=torch.float32, device='cuda'),
    torch.randint(0, 32, [2048, 4], dtype=torch.int64, device='cuda'),
    torch.randn([4, 512, 2880], dtype=torch.float32, device='cuda'),
    torch.randn([2880], dtype=torch.float32, device='cuda'),
    [32, 4, -1, 2880],  # _shape_param_0
    [32, 4, -1],  # _shape_param_1
    [2048, 2880],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
