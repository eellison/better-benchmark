"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([2048, 1024], bf16), T([4, 512, 1024], bf16), T([1024], bf16), T([2048, 1024], bf16), T([1024, 1024], bf16), T([1024, 1024], bf16))"

class Repro(torch.nn.Module):
    def forward(self, mm_188: "bf16[2048, 1024]", add_242: "bf16[4, 512, 1024]", arg300_1: "bf16[1024]", arg301_1: "bf16[2048, 1024]", arg303_1: "bf16[1024, 1024]", arg305_1: "bf16[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_188, [4, 512, 1024]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:333 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_242, reshape_default);  add_242 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg300_1, convert_element_type_default_1);  arg300_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:263 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_1: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 1024])
        permute_default: "bf16[1024, 2048]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:264 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_2: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 1024])
        permute_default_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:265 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_3: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 1024]);  mul_tensor_1 = None
        permute_default_2: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2)


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
