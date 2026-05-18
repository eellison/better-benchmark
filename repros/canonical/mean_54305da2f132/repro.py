"""
Standalone repro captured via capture_hook.
Label: hf_qwen2_0.5b_train
Pattern hash: 54305da2f132
Shape hash: f5951b67
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
    def forward(self, mm_91: "bf16[2048, 896]", _shape_param_0, add_160: "bf16[4, 512, 896]", primals_280: "bf16[896]", _shape_param_1, primals_281: "bf16[896, 896]", primals_283: "bf16[128, 896]", primals_285: "bf16[128, 896]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(mm_91, _shape_param_0);  mm_91 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:308 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "bf16[4, 512, 896]" = torch.ops.aten.add.Tensor(add_160, reshape_default);  add_160 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:260 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 896]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:261 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 896]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(primals_280, convert_element_type_default_1);  primals_280 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "bf16[2048, 896]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default: "bf16[896, 896]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_1: "bf16[896, 128]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_2: "bf16[896, 128]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        return (reshape_default_1, permute_default, permute_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_0
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([896], dtype=torch.bfloat16, device='cuda'),
    [2048, 896],  # _shape_param_1
    torch.randn([896, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 896], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
