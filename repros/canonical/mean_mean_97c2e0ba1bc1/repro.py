"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_infer
Pattern hash: 97c2e0ba1bc1
Shape hash: af2aa253
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 512], f32), T([32, 128, 512], f32), T([512], f32), T([384, 512], f32), T([4096, 512], f32), T([32, 128, 512], f32), T([512], f32), T([384, 512], f32), T([384, 512], f32), S([32, 128, 512]), S([4096, 512]), S([32, 128, 512]), S([4096, 512]), S([4096, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_59: "f32[4096, 512]", embedding_2: "f32[32, 128, 512]", arg83_1: "f32[512]", arg84_1: "f32[384, 512]", mm_55: "f32[4096, 512]", add_58: "f32[32, 128, 512]", arg75_1: "f32[512]", arg85_1: "f32[384, 512]", arg86_1: "f32[384, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        reshape_default: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_59, _shape_param_0);  mm_59 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(embedding_2, reshape_default);  embedding_2 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg83_1, mul_tensor);  arg83_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default: "f32[512, 384]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_2: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_55, _shape_param_2);  mm_55 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_58, reshape_default_2);  add_58 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_2, 2)
        mean_dim_1: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True);  pow_tensor_scalar_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_3: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        rsqrt_default_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, rsqrt_default_1);  add_tensor_2 = rsqrt_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg75_1, mul_tensor_2);  arg75_1 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_3);  _shape_param_3 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_4: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_4);  mul_tensor_3 = _shape_param_4 = None
        permute_default_2: "f32[512, 384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        return (reshape_default_1, permute_default, reshape_default_3, permute_default_1, reshape_default_4, permute_default_2)


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
