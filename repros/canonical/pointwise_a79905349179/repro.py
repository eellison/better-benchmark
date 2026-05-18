"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: a79905349179
Shape hash: 03dd8bb1
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
    def forward(self, bmm_68: "f32[48, 512, 64]", _shape_param_0, bmm_70: "f32[48, 64, 512]", _shape_param_1, bmm_71: "f32[48, 512, 64]", _shape_param_2, _shape_param_3, _shape_param_4, primals_6: "f32[768, 768]", _shape_param_5, _shape_param_6, primals_5: "f32[768, 768]", _shape_param_7, _shape_param_8, primals_4: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, _shape_param_0);  bmm_68 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_1: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_70, _shape_param_1);  bmm_70 = _shape_param_1 = None
        reshape_default_2: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, _shape_param_2);  bmm_71 = _shape_param_2 = None
        permute_default: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_5: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_5);  permute_default_4 = _shape_param_5 = None
        clone_default_1: "f32[4, 512, 768]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_5: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_2: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_7: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_9: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_4, permute_default_3, reshape_default_6, permute_default_6, reshape_default_8, permute_default_9)


def _default_make_inputs():
    return [
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_0
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_1
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_2
    [4, 512, 768],  # _shape_param_3
    [2048, 768],  # _shape_param_4
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_5
    [2048, 768],  # _shape_param_6
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_7
    [2048, 768],  # _shape_param_8
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
