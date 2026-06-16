"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer
Pattern hash: 07c6ea330ab4
Shape hash: 5fa3702b
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
    def forward(self, arg0_1: "bf16[8192, 768]", arg1_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view, arg1_1);  view = arg1_1 = None
        view_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1, [1, 0, 2, 3]);  view_1 = None
        permute_1: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute, [0, 2, 1, 3]);  permute = None
        view_2: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_2);  permute_1 = _shape_param_2 = None
        constant_pad_nd: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_2, _shape_param_3, -1.0);  view_2 = _shape_param_3 = None
        as_strided: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd, _shape_param_4, _shape_param_5);  constant_pad_nd = _shape_param_4 = _shape_param_5 = None
        unsqueeze: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        clone: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze, memory_format = torch.contiguous_format);  unsqueeze = None
        view_3: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone, _shape_param_6);  clone = _shape_param_6 = None
        return view_3



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
