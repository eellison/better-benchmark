"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train_005
Pattern hash: 5eec4697d8ff
Shape hash: 9aa80f9a
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
_shapes_config = "(T([96, 768, 64], f32), T([2359296], f32), T([4718592], i64, gen=Index(2359296)), S([24, 4, 768, 64, 1]), S([2, 12, 1024, 64]), S([1024, 2, 768]), S([768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_44: "f32[96, 768, 64]", full_2: "f32[2359296]", view_19: "i64[4718592]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_44, _shape_param_0);  bmm_44 = _shape_param_0 = None
        squeeze_dim: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_default, 4);  view_default = None
        view_default_1: "f32[4718592]" = torch.ops.aten.view.default(squeeze_dim, [-1]);  squeeze_dim = None
        index_put_default: "f32[2359296]" = torch.ops.aten.index_put.default(full_2, [view_19], view_default_1, True);  full_2 = view_19 = view_default_1 = None
        as_strided_default: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_default, [24, 1536, 64], [98304, 64, 1], 0);  index_put_default = None
        constant_pad_nd_default: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_default, [0, 0, -256, -256]);  as_strided_default = None
        view_default_2: "f32[2, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_default, _shape_param_1);  constant_pad_nd_default = _shape_param_1 = None
        permute_default: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        permute_default_1: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_default, [1, 0, 2, 3]);  permute_default = None
        clone_default: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3: "f32[1024, 2, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0, 1], True)
        view_default_4: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        view_default_5: "f32[2048, 768]" = torch.ops.aten.view.default(view_default_3, _shape_param_4);  view_default_3 = _shape_param_4 = None
        permute_default_2: "f32[768, 2048]" = torch.ops.aten.permute.default(view_default_5, [1, 0]);  view_default_5 = None
        return (view_default_4, permute_default_2)



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
