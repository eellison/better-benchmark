"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph42
Pattern hash: 0b207f294ce4
Shape hash: 24159439
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f16[128, 256008]", arg2_1: "i64[1, 128]", permute: "f16[1024, 256008]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[1, 128, 256008]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type_default: "f32[1, 128, 256008]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        constant_pad_nd_default: "i64[1, 129]" = torch.ops.aten.constant_pad_nd.default(arg2_1, [0, 1], -100.0);  arg2_1 = None
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, 1, 1, 9223372036854775807);  constant_pad_nd_default = None
        view_default_1: "f32[128, 256008]" = torch.ops.aten.view.default(convert_element_type_default, _shape_param_1);  convert_element_type_default = _shape_param_1 = None
        view_default_2: "i64[128]" = torch.ops.aten.view.default(slice_tensor, [-1]);  slice_tensor = None
        amax_default: "f32[128, 1]" = torch.ops.aten.amax.default(view_default_1, [1], True)
        sub_tensor: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(view_default_1, amax_default);  view_default_1 = amax_default = None
        exp_default: "f32[128, 256008]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[128, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[128]" = torch.ops.aten.ne.Scalar(view_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[128]" = torch.ops.aten.where.self(ne_scalar, view_default_2, full_default);  view_default_2 = full_default = None
        unsqueeze_default: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[128, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[128]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[128]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[128]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_1);  sum_default_1 = convert_element_type_default_1 = None
        permute_default: "f16[256008, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div_tensor, permute_default)


def _default_make_inputs():
    return [
    torch.randn([128, 256008], dtype=torch.float16, device='cuda'),
    torch.randint(0, 128, [1, 128], dtype=torch.int64, device='cuda'),
    torch.randn(262152192, dtype=torch.float16, device='cuda').as_strided([1024, 256008], [1, 1024]),  # permute
    [1, 128, 256008],  # _shape_param_0
    [-1, 256008],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
