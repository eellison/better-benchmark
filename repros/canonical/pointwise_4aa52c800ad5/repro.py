"""
Standalone repro captured via capture_hook.
Label: hf_pythia_410m_train
Pattern hash: 4aa52c800ad5
Shape hash: 6c04dd52
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_92: "f16[2048, 3072]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:214 in forward, code: qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_92, _shape_param_0);  addmm_92 = _shape_param_0 = None
        reshape_default_1: "f16[4, 512, 16, 192]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[4, 16, 512, 192]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:215 in forward, code: query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        split_tensor = torch.ops.aten.split.Tensor(permute_default, 64, -1);  permute_default = None
        getitem: "f16[4, 16, 512, 64]" = split_tensor[0]
        getitem_1: "f16[4, 16, 512, 64]" = split_tensor[1]
        getitem_2: "f16[4, 16, 512, 64]" = split_tensor[2];  split_tensor = None
        return (getitem, getitem_1, getitem_2)


def _default_make_inputs():
    return [
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [4, 512, 3072],  # _shape_param_0
    [4, 512, -1, 192],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
