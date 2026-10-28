"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/PLBartForCausalLM/full_graph_000.py
Pattern hash: 27eae3e6b7d0
Shape hash: dde36612
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[50005, 768]", arg1_1: "bf16[16384, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "bf16[768, 50005]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        constant_pad_nd: "bf16[768, 50008]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        mm: "bf16[16384, 50008]" = torch.ops.aten.mm.default(arg1_1, constant_pad_nd);  arg1_1 = constant_pad_nd = None
        slice_1: "bf16[16384, 50005]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -3);  mm = None
        view: "bf16[16, 1024, 50005]" = torch.ops.aten.view.default(slice_1, _shape_param_1);  slice_1 = _shape_param_1 = None
        view_1: "bf16[16384, 50005]" = torch.ops.aten.view.default(view, _shape_param_2);  _shape_param_2 = None
        convert_element_type: "f32[16384, 50005]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        return (view, convert_element_type)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
