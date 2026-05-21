"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer_000
Pattern hash: a942dec89d15
Shape hash: 47a5cbda
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
_shapes_config = "(T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant0: "f32[]"):
        # No stacktrace found for following nodes
        lift_fresh_copy_default: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        mul_tensor: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default, 1);  lift_fresh_copy_default = None
        return mul_tensor



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
