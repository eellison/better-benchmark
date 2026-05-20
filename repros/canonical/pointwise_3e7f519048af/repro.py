"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 3e7f519048af
Shape hash: 9b7fe6ae
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8, 1024, 768], f32), T([768, 768], f32), T([768, 768], f32), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 1024, 768]", primals_2: "f32[768, 768]", primals_4: "f32[768, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_default: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(primals_1, [1, 0, 2]);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        clone_default: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        return (permute_default_1, reshape_default, permute_default_2)


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
