"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=mean, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_18: "f32[2048, 2880]", add_31: "f32[4, 512, 2880]", arg64_1: "f32[2880]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:342 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default: "f32[4, 512, 2880]" = torch.ops.aten.reshape.default(addmm_18, [4, 512, 2880]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:381 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[4, 512, 2880]" = torch.ops.aten.add.Tensor(add_31, reshape_default);  add_31 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:59 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 2880]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:60 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:61 in forward, code: return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama
        mul_tensor_1: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(arg64_1, mul_tensor);  arg64_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:153 in forward, code: hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        reshape_default_1: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_1, [-1, 2880])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:96 in forward, code: hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        reshape_default_2: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_1, [-1, 2880]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:128 in forward, code: hidden_states = hidden_states.repeat(num_experts, 1)
        repeat_default: "f32[65536, 2880]" = torch.ops.aten.repeat.default(reshape_default_2, [32, 1]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:129 in forward, code: hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        reshape_default_3: "f32[32, 2048, 2880]" = torch.ops.aten.reshape.default(repeat_default, [32, -1, 2880]);  repeat_default = None
        return (reshape_default_1, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([2048, 2880], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 2880], dtype=torch.float32, device='cuda'),
    torch.randn([2880], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
