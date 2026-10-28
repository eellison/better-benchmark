"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/GPTNeoForCausalLM/full_graph_000.py
Pattern hash: 48994be71178
Shape hash: 2a2aacfc
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
    def forward(self, arg0_1: "f32[50257, 2048]", arg1_1: "bf16[4096, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[50257, 2048]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        permute: "bf16[2048, 50257]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        constant_pad_nd: "bf16[2048, 50264]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        mm: "bf16[4096, 50264]" = torch.ops.aten.mm.default(arg1_1, constant_pad_nd);  arg1_1 = constant_pad_nd = None
        slice_1: "bf16[4096, 50257]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -7);  mm = None
        view: "bf16[32, 128, 50257]" = torch.ops.aten.view.default(slice_1, _shape_param_1);  slice_1 = _shape_param_1 = None
        convert_element_type_1: "f32[32, 128, 50257]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        view_1: "f32[4096, 50257]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        return (view, view_1)



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
