"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: 26178cdd79b9
Shape hash: daf2bc2d
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
    def forward(self, mm_66: "f32[2048, 768]", _shape_param_0, _shape_param_1, mm_67: "f32[2048, 768]", _shape_param_2, _shape_param_3, primals_95: "f32[768, 768]", _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_66, _shape_param_0);  mm_66 = _shape_param_0 = None
        reshape_default_1: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_67, _shape_param_2);  mm_67 = _shape_param_2 = None
        reshape_default_3: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_3: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_4);  permute_default = _shape_param_4 = None
        clone_default: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        expand_default_1: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None
        clone_default_1: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        return (permute_default_2, reshape_default_4, reshape_default_5)


def _default_make_inputs():
    return [
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_0
    [4, 512, -1, 64],  # _shape_param_1
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_2
    [4, 512, -1, 64],  # _shape_param_3
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_4
    [48, 512, 64],  # _shape_param_5
    [4, 12, 64, 512],  # _shape_param_6
    [48, 64, 512],  # _shape_param_7
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
