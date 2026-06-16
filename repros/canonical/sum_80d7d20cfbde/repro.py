"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 80d7d20cfbde
Shape hash: f85fe078
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
    def forward(self, arg0_1: "bf16[32, 6, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        clone: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        _unsafe_view: "bf16[32, 512, 384]" = torch.ops.aten._unsafe_view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        clone_1: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(_unsafe_view, memory_format = torch.contiguous_format);  _unsafe_view = None
        view: "bf16[16384, 384]" = torch.ops.aten.view.default(clone_1, _shape_param_1);  clone_1 = _shape_param_1 = None
        permute_1: "bf16[384, 16384]" = torch.ops.aten.permute.default(view, [1, 0])
        sum_1: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32)
        view_1: "f32[384]" = torch.ops.aten.view.default(sum_1, _shape_param_2);  sum_1 = _shape_param_2 = None
        convert_element_type: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_1: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view, permute_1, convert_element_type_1)



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
