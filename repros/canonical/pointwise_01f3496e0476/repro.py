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
    def forward(self, add_14: "f32[16, 128, 2560]", getitem_17: "f32[16, 128, 1]", getitem_16: "f32[16, 128, 1]", arg36_1: "f32[2560]", arg37_1: "f32[2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:836 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_14, getitem_17);  add_14 = getitem_17 = None
        add_tensor: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_default: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg36_1);  mul_tensor = arg36_1 = None
        add_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg37_1);  mul_tensor_1 = arg37_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_1: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_2);  _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_3: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_3);  _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_4: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_4);  _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_5: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_5);  _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_6: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_6);  _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_7: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_7);  _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_8: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_8);  _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_9: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_9);  _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_10: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_10);  _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_11: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_11);  _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_12: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_12);  _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_13: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_13);  _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_14: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_14);  _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_15: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_15);  _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_16: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_16);  _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_17: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_17);  _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_18: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_18);  _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_19: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_19);  _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_20: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_20);  _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_21: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_21);  _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_22: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_22);  _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_23: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_23);  _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_24: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_24);  _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_25: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_25);  _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_26: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_26);  _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_27: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_27);  _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_28: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_28);  _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_29: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_29);  _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_30: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_30);  _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_31: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_31);  _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_32: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_32);  _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_33: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_33);  _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_34: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_34);  _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_35: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_35);  _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_36: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_36);  _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_37: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_37);  _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_38: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_38);  _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_39: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_39);  _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_40: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_40);  _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_41: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_41);  _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_42: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_42);  _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_43: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_43);  _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_44: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_44);  _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_45: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_45);  _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:236 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_46: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_46);  _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:237 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_47: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_47);  add_tensor_1 = _shape_param_47 = None
        return (reshape_default, reshape_default_1, reshape_default_2, reshape_default_3, reshape_default_4, reshape_default_5, reshape_default_6, reshape_default_7, reshape_default_8, reshape_default_9, reshape_default_10, reshape_default_11, reshape_default_12, reshape_default_13, reshape_default_14, reshape_default_15, reshape_default_16, reshape_default_17, reshape_default_18, reshape_default_19, reshape_default_20, reshape_default_21, reshape_default_22, reshape_default_23, reshape_default_24, reshape_default_25, reshape_default_26, reshape_default_27, reshape_default_28, reshape_default_29, reshape_default_30, reshape_default_31, reshape_default_32, reshape_default_33, reshape_default_34, reshape_default_35, reshape_default_36, reshape_default_37, reshape_default_38, reshape_default_39, reshape_default_40, reshape_default_41, reshape_default_42, reshape_default_43, reshape_default_44, reshape_default_45, reshape_default_46, reshape_default_47)


def _default_make_inputs():
    return [
    torch.randn([16, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    [2048, 2560],  # _shape_param_0
    [2048, 2560],  # _shape_param_1
    [2048, 2560],  # _shape_param_2
    [2048, 2560],  # _shape_param_3
    [2048, 2560],  # _shape_param_4
    [2048, 2560],  # _shape_param_5
    [2048, 2560],  # _shape_param_6
    [2048, 2560],  # _shape_param_7
    [2048, 2560],  # _shape_param_8
    [2048, 2560],  # _shape_param_9
    [2048, 2560],  # _shape_param_10
    [2048, 2560],  # _shape_param_11
    [2048, 2560],  # _shape_param_12
    [2048, 2560],  # _shape_param_13
    [2048, 2560],  # _shape_param_14
    [2048, 2560],  # _shape_param_15
    [2048, 2560],  # _shape_param_16
    [2048, 2560],  # _shape_param_17
    [2048, 2560],  # _shape_param_18
    [2048, 2560],  # _shape_param_19
    [2048, 2560],  # _shape_param_20
    [2048, 2560],  # _shape_param_21
    [2048, 2560],  # _shape_param_22
    [2048, 2560],  # _shape_param_23
    [2048, 2560],  # _shape_param_24
    [2048, 2560],  # _shape_param_25
    [2048, 2560],  # _shape_param_26
    [2048, 2560],  # _shape_param_27
    [2048, 2560],  # _shape_param_28
    [2048, 2560],  # _shape_param_29
    [2048, 2560],  # _shape_param_30
    [2048, 2560],  # _shape_param_31
    [2048, 2560],  # _shape_param_32
    [2048, 2560],  # _shape_param_33
    [2048, 2560],  # _shape_param_34
    [2048, 2560],  # _shape_param_35
    [2048, 2560],  # _shape_param_36
    [2048, 2560],  # _shape_param_37
    [2048, 2560],  # _shape_param_38
    [2048, 2560],  # _shape_param_39
    [2048, 2560],  # _shape_param_40
    [2048, 2560],  # _shape_param_41
    [2048, 2560],  # _shape_param_42
    [2048, 2560],  # _shape_param_43
    [2048, 2560],  # _shape_param_44
    [2048, 2560],  # _shape_param_45
    [2048, 2560],  # _shape_param_46
    [2048, 2560],  # _shape_param_47
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
