"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=mean, ranges=['8', '1024', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_35: "f32[8192, 512]", add_31: "f32[8, 1024, 512]", arg51_1: "f32[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:297 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_35, [8, 1024, 512]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:344 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_31, reshape_default);  add_31 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:255 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg51_1, mul_tensor);  arg51_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_1: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_2: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_3: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_4: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_5: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_6: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_7: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_8: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_9: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_10: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_11: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_12: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 512]);  mul_tensor_1 = None
        return (reshape_default_1, reshape_default_2, reshape_default_3, reshape_default_4, reshape_default_5, reshape_default_6, reshape_default_7, reshape_default_8, reshape_default_9, reshape_default_10, reshape_default_11, reshape_default_12)


def _default_make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
