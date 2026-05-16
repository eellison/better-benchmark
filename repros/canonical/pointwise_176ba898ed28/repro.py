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
    def forward(self, _tensor_constant23: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:125 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
        mul_tensor: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default, 1);  lift_fresh_copy_default = None
        return mul_tensor


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cpu'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
