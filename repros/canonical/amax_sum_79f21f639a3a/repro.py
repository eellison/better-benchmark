"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer_000
Pattern hash: 79f21f639a3a
Shape hash: 2810a40e
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
_shapes_config = "(T([16384, 20005], f32), S([128, 128, 20005]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_73: "f32[16384, 20005]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[128, 128, 20005]" = torch.ops.aten.view.default(addmm_73, _shape_param_0);  addmm_73 = _shape_param_0 = None
        amax_default: "f32[128, 128, 1]" = torch.ops.aten.amax.default(view_default, [-1], True)
        sub_tensor: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(view_default, amax_default);  view_default = amax_default = None
        exp_default: "f32[128, 128, 20005]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[128, 128, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        return sub_tensor_1



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
