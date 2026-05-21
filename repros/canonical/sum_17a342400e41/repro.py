"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 17a342400e41
Shape hash: 88016393
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
_shapes_config = "(T([512, 512, 512], f32), T([8, 64, 512, 512], f32), S([8, 64, 512, 512]), S([512, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_45: "f32[512, 512, 512]", arg20_1: "f32[8, 64, 512, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_45, _shape_param_0);  bmm_45 = _shape_param_0 = None
        mul_tensor: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(view_default, arg20_1);  view_default = None
        sum_dim_int_list: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(arg20_1);  arg20_1 = None
        fma_default: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        view_default_1: "f32[512, 512, 512]" = torch.ops.aten.view.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        return view_default_1



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
