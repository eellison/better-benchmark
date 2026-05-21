"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_GPT2_large_train
Pattern hash: cbc38fce0bb9
Shape hash: 6a69677a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 3840], f32), S([4, 512, 3840]), S([4, 512, -1, 64]), S([4, 512, -1, 64]), S([4, 512, -1, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_108: "f32[2048, 3840]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default: "f32[4, 512, 3840]" = torch.ops.aten.reshape.default(addmm_108, _shape_param_0);  addmm_108 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default, 1280, 2);  reshape_default = None
        getitem: "f32[4, 512, 1280]" = split_tensor[0]
        getitem_1: "f32[4, 512, 1280]" = split_tensor[1]
        getitem_2: "f32[4, 512, 1280]" = split_tensor[2];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        reshape_default_1: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_1, _shape_param_1);  getitem_1 = _shape_param_1 = None
        permute_default: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        reshape_default_2: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem_2, _shape_param_2);  getitem_2 = _shape_param_2 = None
        permute_default_1: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        reshape_default_3: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(getitem, _shape_param_3);  getitem = _shape_param_3 = None
        permute_default_2: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        return (permute_default_1, permute_default, permute_default_2)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
