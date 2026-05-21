"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train
Pattern hash: ff702a028ea6
Shape hash: 625ea58c
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
_shapes_config = "(T([2048, 768], f32), T([2048, 768], f32), T([2048, 768], f32), T([2, 1024, 768], f32), S([1024, 2, 768]), S([1024, 2, 768]), S([1024, 2, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_187: "f32[2048, 768]", mm_189: "f32[2048, 768]", mm_191: "f32[2048, 768]", mul_497: "f32[2, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_187, _shape_param_0);  mm_187 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default_1: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_189, _shape_param_1);  mm_189 = _shape_param_1 = None
        add_tensor: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default_2: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_191, _shape_param_2);  mm_191 = _shape_param_2 = None
        add_tensor_1: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_default: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        add_tensor_2: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_497, permute_default);  mul_497 = permute_default = None
        return add_tensor_2



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
