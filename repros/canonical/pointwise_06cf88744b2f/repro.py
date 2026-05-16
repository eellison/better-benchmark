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
    def forward(self, addmm: "f32[16384, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:141 in forward, code: embeddings = self.projection(embeddings)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm, [32, 512, 768]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:180 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_default: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.complex64);  reshape_default = None
        return convert_element_type_default


def _default_make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
