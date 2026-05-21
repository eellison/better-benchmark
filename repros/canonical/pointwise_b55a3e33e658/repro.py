"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer_000
Pattern hash: b55a3e33e658
Shape hash: 82a099dc
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
_shapes_config = "(T([12000, 384], f16), S([8, 1500, 384]), S([8, 1500, -1, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_15: "f16[12000, 384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[8, 1500, 384]" = torch.ops.aten.view.default(addmm_15, _shape_param_0);  addmm_15 = _shape_param_0 = None
        mul_tensor: "f16[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_default, 0.125);  view_default = None
        view_default_1: "f16[8, 1500, 6, 64]" = torch.ops.aten.view.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        permute_default: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        clone_default: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        return clone_default



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
