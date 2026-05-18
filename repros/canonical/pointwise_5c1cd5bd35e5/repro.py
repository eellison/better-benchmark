"""
Standalone repro captured via capture_hook.
Label: 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_training
Pattern hash: 5c1cd5bd35e5
Shape hash: e6fa2778
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
    def forward(self, bmm_2: "f32[256, 128, 80]", _shape_param_0, bmm_4: "f32[256, 80, 128]", _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, primals_20: "f32[2560, 2560]", _shape_param_5, primals_18: "f32[2560, 2560]", getitem_12: "f32[8, 32, 128, 80]", _shape_param_6, getitem_11: "f32[8, 32, 128, 80]", _shape_param_7, _shape_param_8, primals_8: "f32[2560, 2560]", _shape_param_9, primals_6: "f32[2560, 2560]", getitem_10: "f32[8, 32, 128, 80]", _shape_param_10, _shape_param_11, primals_4: "f32[2560, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[8, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_2, _shape_param_0);  bmm_2 = _shape_param_0 = None
        reshape_default_1: "f32[8, 32, 80, 128]" = torch.ops.aten.reshape.default(bmm_4, _shape_param_1);  bmm_4 = _shape_param_1 = None
        mul_scalar: "f32[8, 32, 80, 128]" = torch.ops.aten.mul.Scalar(reshape_default_1, 0.334370152488211);  reshape_default_1 = None
        permute_default: "f32[8, 32, 128, 80]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[8, 128, 32, 80]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 128, 32, 80]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[8, 128, 32, 80]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_3: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_3);  permute_default_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_4: "f32[1024, 2560]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        permute_default_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_default_4: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        clone_default_1: "f32[8, 128, 2560]" = torch.ops.aten.clone.default(reshape_default_3, memory_format = torch.contiguous_format);  reshape_default_3 = None
        reshape_default_5: "f32[1024, 2560]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        permute_default_5: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_6: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_default_7: "f32[8, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
        reshape_default_6: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_7, _shape_param_6);  permute_default_7 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_default_8: "f32[8, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
        reshape_default_7: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_7);  permute_default_8 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_8: "f32[1024, 2560]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_8);  reshape_default_6 = _shape_param_8 = None
        permute_default_9: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_10: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_9: "f32[1024, 2560]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_9);  reshape_default_7 = _shape_param_9 = None
        permute_default_11: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_12: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_11, [1, 0]);  permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_13: "f32[8, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        reshape_default_10: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_13, _shape_param_10);  permute_default_13 = _shape_param_10 = None
        reshape_default_11: "f32[1024, 2560]" = torch.ops.aten.reshape.default(reshape_default_10, _shape_param_11);  reshape_default_10 = _shape_param_11 = None
        permute_default_14: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_15: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_14, [1, 0]);  permute_default_14 = None
        return (reshape_default_4, permute_default_4, reshape_default_5, permute_default_6, reshape_default_8, permute_default_10, reshape_default_9, permute_default_12, reshape_default_11, permute_default_15)


def _default_make_inputs():
    return [
    torch.randn([256, 128, 80], dtype=torch.float32, device='cuda'),
    [8, 32, 128, 80],  # _shape_param_0
    torch.randn([256, 80, 128], dtype=torch.float32, device='cuda'),
    [8, 32, 80, 128],  # _shape_param_1
    [8, 128, 2560],  # _shape_param_2
    [8, 128, 2560],  # _shape_param_3
    [1024, 2560],  # _shape_param_4
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [1024, 2560],  # _shape_param_5
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn(2621440, dtype=torch.float32, device='cuda').as_strided([8, 32, 128, 80], [327680, 80, 2560, 1]),  # getitem_12
    [8, 128, 2560],  # _shape_param_6
    torch.randn(2621440, dtype=torch.float32, device='cuda').as_strided([8, 32, 128, 80], [327680, 80, 2560, 1]),  # getitem_11
    [8, 128, 2560],  # _shape_param_7
    [1024, 2560],  # _shape_param_8
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [1024, 2560],  # _shape_param_9
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn(2621440, dtype=torch.float32, device='cuda').as_strided([8, 32, 128, 80], [327680, 80, 2560, 1]),  # getitem_10
    [8, 128, 2560],  # _shape_param_10
    [1024, 2560],  # _shape_param_11
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
