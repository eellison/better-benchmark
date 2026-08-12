"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train
Pattern hash: 045f4feb3853
Shape hash: e6f344ac
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        permute: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type, [0, 2, 1, 3]);  convert_element_type = None
        clone: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute_1)



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
