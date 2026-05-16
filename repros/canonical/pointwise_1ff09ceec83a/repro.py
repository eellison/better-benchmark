"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg68_1: "f32[32, 5760]", bmm_14: "f32[32, 2048, 5760]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:130 in forward, code: gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
        unsqueeze_default: "f32[32, 1, 5760]" = torch.ops.aten.unsqueeze.default(arg68_1, 1);  arg68_1 = None
        add_tensor: "f32[32, 2048, 5760]" = torch.ops.aten.add.Tensor(bmm_14, unsqueeze_default);  bmm_14 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:131 in forward, code: gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        slice_tensor: "f32[32, 2048, 2880]" = torch.ops.aten.slice.Tensor(add_tensor, 2, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:133 in forward, code: up = up.clamp(min=-self.limit, max=self.limit)
        clamp_min_default: "f32[32, 2048, 2880]" = torch.ops.aten.clamp_min.default(slice_tensor, -7.0);  slice_tensor = None
        clamp_max_default: "f32[32, 2048, 2880]" = torch.ops.aten.clamp_max.default(clamp_min_default, 7.0);  clamp_min_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:135 in forward, code: next_states = torch.bmm(((up + 1) * glu), self.down_proj)
        add_tensor_1: "f32[32, 2048, 2880]" = torch.ops.aten.add.Tensor(clamp_max_default, 1);  clamp_max_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:131 in forward, code: gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        slice_tensor_1: "f32[32, 2048, 2880]" = torch.ops.aten.slice.Tensor(add_tensor, 2, 0, 9223372036854775807, 2);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:132 in forward, code: gate = gate.clamp(min=None, max=self.limit)
        clamp_max_default_1: "f32[32, 2048, 2880]" = torch.ops.aten.clamp_max.default(slice_tensor_1, 7.0);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:134 in forward, code: glu = gate * torch.sigmoid(gate * self.alpha)
        mul_tensor: "f32[32, 2048, 2880]" = torch.ops.aten.mul.Tensor(clamp_max_default_1, 1.702)
        sigmoid_default: "f32[32, 2048, 2880]" = torch.ops.aten.sigmoid.default(mul_tensor);  mul_tensor = None
        mul_tensor_1: "f32[32, 2048, 2880]" = torch.ops.aten.mul.Tensor(clamp_max_default_1, sigmoid_default);  clamp_max_default_1 = sigmoid_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:135 in forward, code: next_states = torch.bmm(((up + 1) * glu), self.down_proj)
        mul_tensor_2: "f32[32, 2048, 2880]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_1);  add_tensor_1 = mul_tensor_1 = None
        return mul_tensor_2


def _default_make_inputs():
    return [
    torch.randn([32, 5760], dtype=torch.float32, device='cuda'),
    torch.randn([32, 2048, 5760], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
