"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: a06452a815ae
Shape hash: 3d74fff8
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_137: "f32[4096, 384]", mm_138: "f32[4096, 384]", primals_185: "f32[384, 512]", mul_143: "f32[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_137, _shape_param_0);  mm_137 = _shape_param_0 = None
        reshape_default_1: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_138, _shape_param_2);  mm_138 = _shape_param_2 = None
        reshape_default_3: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[512, 384]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        reshape_default_4: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_4);  mul_143 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_3: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_5: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        expand_default_1: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_default_3, _shape_param_7);  permute_default_3 = _shape_param_7 = None
        clone_default_1: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_6: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_8);  clone_default_1 = _shape_param_8 = None
        return (permute_default_2, reshape_default_4, reshape_default_5, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 512], dtype=torch.float32, device='cuda'),
    [32, 128, 384],  # _shape_param_0
    [32, 128, -1, 64],  # _shape_param_1
    [32, 128, 384],  # _shape_param_2
    [32, 128, -1, 64],  # _shape_param_3
    [4096, 512],  # _shape_param_4
    [32, 6, 128, 64],  # _shape_param_5
    [192, 128, 64],  # _shape_param_6
    [32, 6, 64, 128],  # _shape_param_7
    [192, 64, 128],  # _shape_param_8
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
