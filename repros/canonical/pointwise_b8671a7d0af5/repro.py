"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[32768, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:787 in forward, code: hidden_states = self.embeddings_project(hidden_states)
        reshape_default: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:239 in forward, code: query_layer = self.query(hidden_states)
        reshape_default_1: "f32[32768, 256]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:263 in forward, code: key_layer = self.key(current_states)
        reshape_default_2: "f32[32768, 256]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_2);  _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:267 in forward, code: value_layer = self.value(current_states)
        reshape_default_3: "f32[32768, 256]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_3);  reshape_default = _shape_param_3 = None
        return (reshape_default_1, reshape_default_2, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    [64, 512, 256],  # _shape_param_0
    [32768, 256],  # _shape_param_1
    [32768, 256],  # _shape_param_2
    [32768, 256],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
