"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: 82d023e49f62
Shape hash: f07fc870
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 49, 49], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([169, 16], f32), S([512, 16, 49, 49]), S([2401, 16]), S([8192, 49, 49]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_77: "f32[8192, 49, 49]", arg234_1: "f32[512, 16, 49, 49]", arg38_1: "i64[49, 49]", full_1: "f32[169, 16]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_77, _shape_param_0);  bmm_77 = _shape_param_0 = None
        mul_tensor: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_default, arg234_1);  view_default = None
        sum_dim_int_list: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(arg234_1);  arg234_1 = None
        fma_default: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        sum_dim_int_list_1: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_default, [0], True)
        squeeze_dim: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        permute_default: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None
        view_default_1: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        view_default_2: "i64[2401]" = torch.ops.aten.view.default(arg38_1, [-1]);  arg38_1 = None
        index_put_default: "f32[169, 16]" = torch.ops.aten.index_put.default(full_1, [view_default_2], view_default_1, True);  full_1 = view_default_2 = view_default_1 = None
        view_default_3: "f32[8192, 49, 49]" = torch.ops.aten.view.default(fma_default, _shape_param_2);  fma_default = _shape_param_2 = None
        return (index_put_default, view_default_3)

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
