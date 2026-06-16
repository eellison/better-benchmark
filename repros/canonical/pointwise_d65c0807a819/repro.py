"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: d65c0807a819
Shape hash: 53c69788
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "bf16[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        view: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        add: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view, convert_element_type);  view = convert_element_type = None
        view_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1, [1, 0, 2, 3]);  view_1 = None
        permute_1: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute, [0, 2, 1, 3]);  permute = None
        view_2: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_2);  permute_1 = _shape_param_2 = None
        view_3: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        as_strided: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_3, _shape_param_4, _shape_param_5);  view_3 = _shape_param_4 = _shape_param_5 = None
        unsqueeze: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        permute_2: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 4, 2, 3]);  unsqueeze = None
        permute_3: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_2, [0, 1, 4, 3, 2]);  permute_2 = None
        clone: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_4: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone, _shape_param_6);  clone = _shape_param_6 = None
        permute_4: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1])
        return (view_4, permute_4)



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
