"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[32768, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:787 in forward, code: hidden_states = self.embeddings_project(hidden_states)
        reshape_default: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm, [64, 512, 256]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:239 in forward, code: query_layer = self.query(hidden_states)
        reshape_default_1: "f32[32768, 256]" = torch.ops.aten.reshape.default(reshape_default, [32768, 256])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:263 in forward, code: key_layer = self.key(current_states)
        reshape_default_2: "f32[32768, 256]" = torch.ops.aten.reshape.default(reshape_default, [32768, 256])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:267 in forward, code: value_layer = self.value(current_states)
        reshape_default_3: "f32[32768, 256]" = torch.ops.aten.reshape.default(reshape_default, [32768, 256]);  reshape_default = None
        return (reshape_default_1, reshape_default_2, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
