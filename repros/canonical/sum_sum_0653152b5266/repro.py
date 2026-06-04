"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train_005
Pattern hash: 0653152b5266
Shape hash: 0763094e
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([72, 64, 512], f32), T([72, 512, 64], f32), S([24, 3, 64, 512, 1]), S([24, 3, 512, 64, 1]), S([2359296]), S([2359296]), S([24, 1024, 64]), S([2, 12, 1024, 64]), S([1024, 2, 768]), S([24, 1024, 64]), S([2, 12, 1024, 64]), S([1024, 2, 768]), S([768]), S([2048, 768]), S([768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_2: "f32[72, 64, 512]", bmm_3: "f32[72, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        view_default: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_2, _shape_param_0);  bmm_2 = _shape_param_0 = None
        permute_default: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_default, [0, 1, 4, 3, 2]);  view_default = None
        view_default_1: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_3, _shape_param_1);  bmm_3 = _shape_param_1 = None
        permute_default_1: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_default, [0, 1, 3, 4, 2]);  permute_default = None
        squeeze_dim: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_default_1, 4);  permute_default_1 = None
        squeeze_dim_1: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_default_1, 4);  view_default_1 = None
        full_default: "f32[1572864]" = torch.ops.aten.full.default([1572864], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[1572864]" = torch.ops.prims.iota.default(1572864, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        as_strided_default: "i64[24, 3, 512, 64]" = torch.ops.aten.as_strided.default(iota_default, [24, 3, 512, 64], [64, 393216, 1536, 1], 0);  iota_default = None
        clone_default: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        view_default_2: "f32[2359296]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        clone_default_1: "i64[24, 3, 512, 64]" = torch.ops.aten.clone.default(as_strided_default, memory_format = torch.contiguous_format);  as_strided_default = None
        view_default_3: "i64[2359296]" = torch.ops.aten.view.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        index_put_default: "f32[1572864]" = torch.ops.aten.index_put.default(full_default, [view_default_3], view_default_2, True);  view_default_2 = None
        view_default_4: "f32[2359296]" = torch.ops.aten.view.default(squeeze_dim_1, [-1]);  squeeze_dim_1 = None
        index_put_default_1: "f32[1572864]" = torch.ops.aten.index_put.default(full_default, [view_default_3], view_default_4, True);  full_default = view_default_3 = view_default_4 = None
        as_strided_default_1: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default_1, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_default_1 = None
        view_default_5: "f32[24, 1024, 64]" = torch.ops.aten.view.default(as_strided_default_1, _shape_param_4);  as_strided_default_1 = _shape_param_4 = None
        view_default_6: "f32[2, 12, 1024, 64]" = torch.ops.aten.view.default(view_default_5, _shape_param_5);  view_default_5 = _shape_param_5 = None
        permute_default_2: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        permute_default_3: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_default_2, [1, 0, 2, 3]);  permute_default_2 = None
        view_default_7: "f32[1024, 2, 768]" = torch.ops.aten.view.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None
        div_tensor: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_default_7, 8.0);  view_default_7 = None
        as_strided_default_2: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_default = None
        view_default_8: "f32[24, 1024, 64]" = torch.ops.aten.view.default(as_strided_default_2, _shape_param_7);  as_strided_default_2 = _shape_param_7 = None
        view_default_9: "f32[2, 12, 1024, 64]" = torch.ops.aten.view.default(view_default_8, _shape_param_8);  view_default_8 = _shape_param_8 = None
        permute_default_4: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_9, [0, 2, 1, 3]);  view_default_9 = None
        permute_default_5: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_default_4, [1, 0, 2, 3]);  permute_default_4 = None
        view_default_10: "f32[1024, 2, 768]" = torch.ops.aten.view.default(permute_default_5, _shape_param_9);  permute_default_5 = _shape_param_9 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_10, [0, 1], True)
        view_default_11: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_10);  sum_dim_int_list = _shape_param_10 = None
        view_default_12: "f32[2048, 768]" = torch.ops.aten.view.default(view_default_10, _shape_param_11);  view_default_10 = _shape_param_11 = None
        permute_default_6: "f32[768, 2048]" = torch.ops.aten.permute.default(view_default_12, [1, 0]);  view_default_12 = None
        sum_dim_int_list_1: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_tensor, [0, 1], True)
        view_default_13: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_12);  sum_dim_int_list_1 = _shape_param_12 = None
        view_default_14: "f32[2048, 768]" = torch.ops.aten.view.default(div_tensor, _shape_param_13);  div_tensor = _shape_param_13 = None
        permute_default_7: "f32[768, 2048]" = torch.ops.aten.permute.default(view_default_14, [1, 0]);  view_default_14 = None
        return (view_default_11, permute_default_6, view_default_13, permute_default_7)

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
