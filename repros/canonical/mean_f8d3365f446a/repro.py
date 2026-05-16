"""
Standalone repro captured via capture_hook.
Label: t5_base
Pattern hash: f8d3365f446a
Shape hash: 634dcaae
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_191: "f32[512, 768]", _shape_param_0, add_170: "f32[4, 128, 768]", arg259_1: "f16[768]", _shape_param_1, arg1_1: "f16[32128, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(mm_191, _shape_param_0);  mm_191 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_tensor: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_170, reshape_default);  add_170 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 128, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_default: "f16[4, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f16[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg259_1, convert_element_type_default);  arg259_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_tensor_2: "f16[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 0.03608439182435161);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        reshape_default_1: "f16[512, 768]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f16[768, 32128]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_0
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    [512, 768],  # _shape_param_1
    torch.randn([32128, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
