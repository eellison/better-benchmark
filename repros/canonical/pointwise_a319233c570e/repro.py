"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_training
Pattern hash: a319233c570e
Shape hash: 67f99f0e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_15: "f32[48, 128, 64]", _shape_param_0, _shape_param_1, primals_71: "f32[512, 384]", _shape_param_2, mm_56: "f32[1024, 384]", _shape_param_3, _shape_param_4, mm_57: "f32[1024, 384]", _shape_param_5, _shape_param_6, primals_81: "f32[384, 512]", _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_15, _shape_param_0);  bmm_15 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        reshape_default_1: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        permute_default_1: "f32[384, 512]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        reshape_default_2: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(mm_56, _shape_param_3);  mm_56 = _shape_param_3 = None
        reshape_default_4: "f32[8, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_2: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_5: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(mm_57, _shape_param_5);  mm_57 = _shape_param_5 = None
        reshape_default_6: "f32[8, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_6);  reshape_default_5 = _shape_param_6 = None
        permute_default_3: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[512, 384]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_5: "f32[8, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_3, [0, 1, 3, 2]);  permute_default_3 = None
        expand_default: "f32[8, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_7);  permute_default_2 = _shape_param_7 = None
        clone_default_1: "f32[8, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_7: "f32[48, 128, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_8);  clone_default_1 = _shape_param_8 = None
        expand_default_1: "f32[8, 6, 64, 128]" = torch.ops.aten.expand.default(permute_default_5, _shape_param_9);  permute_default_5 = _shape_param_9 = None
        clone_default_2: "f32[8, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_8: "f32[48, 64, 128]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_10);  clone_default_2 = _shape_param_10 = None
        return (permute_default_1, reshape_default_2, permute_default_4, reshape_default_7, reshape_default_8)


def _default_make_inputs():
    return [
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_0
    [8, 128, -1],  # _shape_param_1
    torch.randn([512, 384], dtype=torch.float32, device='cuda'),
    [1024, 384],  # _shape_param_2
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_3
    [8, 128, -1, 64],  # _shape_param_4
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_5
    [8, 128, -1, 64],  # _shape_param_6
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_7
    [48, 128, 64],  # _shape_param_8
    [8, 6, 64, 128],  # _shape_param_9
    [48, 64, 128],  # _shape_param_10
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
