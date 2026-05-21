"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_005
Pattern hash: e16a4b148154
Shape hash: e98f6a22
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
_shapes_config = "(T([288, 512, 64], f32), T([6291456], f32), T([9437184], i64, gen=Index(6291456)), S([96, 3, 512, 64, 1]), S([96, 1024, 64]), S([8, 12, 1024, 64]), S([1024, 8, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_47: "f32[288, 512, 64]", full_15: "f32[6291456]", view_38: "i64[9437184]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_47, _shape_param_0);  bmm_47 = _shape_param_0 = None
        squeeze_dim: "f32[96, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_default, 4);  view_default = None
        view_default_1: "f32[9437184]" = torch.ops.aten.view.default(squeeze_dim, [-1]);  squeeze_dim = None
        index_put_default: "f32[6291456]" = torch.ops.aten.index_put.default(full_15, [view_38], view_default_1, True);  full_15 = view_38 = view_default_1 = None
        as_strided_default: "f32[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_put_default = None
        view_default_2: "f32[96, 1024, 64]" = torch.ops.aten.view.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        view_default_3: "f32[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_default_2, _shape_param_2);  view_default_2 = _shape_param_2 = None
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        permute_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_default, [1, 0, 2, 3]);  permute_default = None
        view_default_4: "f32[1024, 8, 768]" = torch.ops.aten.view.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        div_tensor: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(view_default_4, 8.0);  view_default_4 = None
        view_default_5: "f32[8192, 768]" = torch.ops.aten.view.default(div_tensor, _shape_param_4);  div_tensor = _shape_param_4 = None
        return view_default_5



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
