"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 52fe1925ccda
Shape hash: 741cfc27
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
    def forward(self, arg0_1: "f32[20005, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[20005, 768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        permute: "bf16[768, 20005]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        constant_pad_nd: "bf16[768, 20008]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_0);  _shape_param_0 = None
        permute_1: "bf16[20005, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (constant_pad_nd, permute_1)



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
