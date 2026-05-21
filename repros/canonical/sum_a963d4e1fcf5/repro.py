"""
Standalone repro captured via capture_hook.
Label: torchbench_tts_angular_infer
Pattern hash: a963d4e1fcf5
Shape hash: aab07da7
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
_shapes_config = "(T([64, 50, 256], f32), S([64, 256]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 50, 256]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/tts_angular/model.py:75 in torch_dynamo_resume_in_forward_at_73, code: d = torch.nn.functional.normalize(d[:, -1], p=2, dim=1)
        select_int: "f32[64, 256]" = torch.ops.aten.select.int(arg0_1, 1, -1);  arg0_1 = None
        pow_tensor_scalar: "f32[64, 256]" = torch.ops.aten.pow.Tensor_Scalar(select_int, 2)
        sum_dim_int_list: "f32[64, 1]" = torch.ops.aten.sum.dim_IntList(pow_tensor_scalar, [1], True);  pow_tensor_scalar = None
        pow_tensor_scalar_1: "f32[64, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_dim_int_list, 0.5);  sum_dim_int_list = None
        clamp_min_default: "f32[64, 1]" = torch.ops.aten.clamp_min.default(pow_tensor_scalar_1, 1e-12);  pow_tensor_scalar_1 = None
        expand_default: "f32[64, 256]" = torch.ops.aten.expand.default(clamp_min_default, _shape_param_0);  clamp_min_default = _shape_param_0 = None
        div_tensor: "f32[64, 256]" = torch.ops.aten.div.Tensor(select_int, expand_default);  select_int = expand_default = None
        return div_tensor



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
