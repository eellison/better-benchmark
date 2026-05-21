"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: fe6cc00e76ae
Shape hash: 5e9db534
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
_shapes_config = "(T([128, 32, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([128, 32, 49, 49], f32), T([49, 49], i64, gen=Index(169)), S([2401, 32]), S([2401, 32]))"

class Repro(torch.nn.Module):
    def forward(self, fma: "f32[128, 32, 49, 49]", arg173_1: "i64[49, 49]", fma_1: "f32[128, 32, 49, 49]", arg166_1: "i64[49, 49]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma, [0], True);  fma = None
        squeeze_dim: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 0);  sum_dim_int_list = None
        permute_default: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None
        view_default: "f32[2401, 32]" = torch.ops.aten.view.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        full_default: "f32[169, 32]" = torch.ops.aten.full.default([169, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default_1: "i64[2401]" = torch.ops.aten.view.default(arg173_1, [-1]);  arg173_1 = None
        index_put_default: "f32[169, 32]" = torch.ops.aten.index_put.default(full_default, [view_default_1], view_default, True);  view_default_1 = view_default = None
        sum_dim_int_list_1: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_1, [0], True);  fma_1 = None
        squeeze_dim_1: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        permute_default_1: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None
        view_default_2: "f32[2401, 32]" = torch.ops.aten.view.default(permute_default_1, _shape_param_1);  permute_default_1 = _shape_param_1 = None
        view_default_3: "i64[2401]" = torch.ops.aten.view.default(arg166_1, [-1]);  arg166_1 = None
        index_put_default_1: "f32[169, 32]" = torch.ops.aten.index_put.default(full_default, [view_default_3], view_default_2, True);  full_default = view_default_3 = view_default_2 = None
        return (index_put_default, index_put_default_1)



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
