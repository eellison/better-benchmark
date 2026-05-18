"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_74: "f32[64, 128, 1024]", getitem_97: "f32[64, 128, 1]", getitem_96: "f32[64, 128, 1]", arg196_1: "f32[1024]", arg197_1: "f32[1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:898 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_74, getitem_97);  add_74 = getitem_97 = None
        add_tensor: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_default: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg196_1);  mul_tensor = arg196_1 = None
        add_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg197_1);  mul_tensor_1 = arg197_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_3: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_4: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_5: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_6: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_7: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_8: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_9: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_10: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_11: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_12: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_13: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_14: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_15: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_16: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_17: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_18: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_19: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_20: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_21: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_22: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_23: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, [8192, 1024]);  add_tensor_1 = None
        return (reshape_default, reshape_default_1, reshape_default_2, reshape_default_3, reshape_default_4, reshape_default_5, reshape_default_6, reshape_default_7, reshape_default_8, reshape_default_9, reshape_default_10, reshape_default_11, reshape_default_12, reshape_default_13, reshape_default_14, reshape_default_15, reshape_default_16, reshape_default_17, reshape_default_18, reshape_default_19, reshape_default_20, reshape_default_21, reshape_default_22, reshape_default_23)


def _default_make_inputs():
    return [
    torch.randn([64, 128, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
