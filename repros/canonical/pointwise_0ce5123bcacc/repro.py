"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_training
Pattern hash: 0ce5123bcacc
Shape hash: c80337f5
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
    def forward(self, view_76: "f32[4096, 50257]", getitem_88: "f32[8, 12, 512, 64]", _shape_param_0, getitem_90: "f32[8, 12, 512, 64]", _shape_param_1, getitem_89: "f32[8, 12, 512, 64]", _shape_param_2, _shape_param_3, primals_7: "f32[768, 2304]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "f32[50257, 4096]" = torch.ops.aten.permute.default(view_76, [1, 0]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_default_1: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_88, [0, 2, 1, 3]);  getitem_88 = None
        reshape_default: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_0);  permute_default_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_default_2: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        reshape_default_1: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_1);  permute_default_2 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_default_3: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_89, [0, 2, 1, 3]);  getitem_89 = None
        reshape_default_2: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_2);  permute_default_3 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_default: "f32[8, 512, 2304]" = torch.ops.aten.cat.default([reshape_default, reshape_default_2, reshape_default_1], 2);  reshape_default = reshape_default_2 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default_3: "f32[4096, 2304]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_4: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        return (permute_default, reshape_default_3, permute_default_4)


def _default_make_inputs():
    return [
    torch.randn([4096, 50257], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_88
    [8, 512, 768],  # _shape_param_0
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_90
    [8, 512, 768],  # _shape_param_1
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_89
    [8, 512, 768],  # _shape_param_2
    [4096, 2304],  # _shape_param_3
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
