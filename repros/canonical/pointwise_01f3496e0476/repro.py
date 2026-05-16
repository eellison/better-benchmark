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
    def forward(self, add_14: "f32[16, 128, 2560]", getitem_17: "f32[16, 128, 1]", getitem_16: "f32[16, 128, 1]", arg36_1: "f32[2560]", arg37_1: "f32[2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:836 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_14, getitem_17);  add_14 = getitem_17 = None
        add_tensor: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_default: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg36_1);  mul_tensor = arg36_1 = None
        add_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg37_1);  mul_tensor_1 = arg37_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_1: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_3: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_4: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_5: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_6: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_7: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_8: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_9: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_10: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_11: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_12: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_13: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_14: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_15: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_16: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_17: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_18: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_19: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_20: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_21: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_22: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_23: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_24: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_25: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_26: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_27: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_28: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_29: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_30: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_31: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_32: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_33: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_34: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_35: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_36: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_37: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_38: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_39: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_40: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_41: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_42: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_43: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_44: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_45: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_46: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_47: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, [2048, 2560]);  add_tensor_1 = None
        return (reshape_default, reshape_default_1, reshape_default_2, reshape_default_3, reshape_default_4, reshape_default_5, reshape_default_6, reshape_default_7, reshape_default_8, reshape_default_9, reshape_default_10, reshape_default_11, reshape_default_12, reshape_default_13, reshape_default_14, reshape_default_15, reshape_default_16, reshape_default_17, reshape_default_18, reshape_default_19, reshape_default_20, reshape_default_21, reshape_default_22, reshape_default_23, reshape_default_24, reshape_default_25, reshape_default_26, reshape_default_27, reshape_default_28, reshape_default_29, reshape_default_30, reshape_default_31, reshape_default_32, reshape_default_33, reshape_default_34, reshape_default_35, reshape_default_36, reshape_default_37, reshape_default_38, reshape_default_39, reshape_default_40, reshape_default_41, reshape_default_42, reshape_default_43, reshape_default_44, reshape_default_45, reshape_default_46, reshape_default_47)


def _default_make_inputs():
    return [
    torch.randn([16, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
