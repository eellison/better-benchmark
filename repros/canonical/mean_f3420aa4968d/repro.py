"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_infer
Pattern hash: f3420aa4968d
Shape hash: e42d52b1
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 512], f32), T([8, 1024, 512], f32), T([512], f32), T([32128, 512], f32), S([8, 1024, 512]), S([8192, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_95: "f32[8192, 512]", add_92: "f32[8, 1024, 512]", arg132_1: "f32[512]", arg1_1: "f32[32128, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_95, _shape_param_0);  mm_95 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_92, reshape_default);  add_92 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg132_1, mul_tensor);  arg132_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 0.04419417382415922);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        reshape_default_1: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[512, 32128]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        return (reshape_default_1, permute_default)


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
