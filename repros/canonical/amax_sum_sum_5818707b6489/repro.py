"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train_000
Pattern hash: 5818707b6489
Shape hash: a7a0c2ca
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
_shapes_config = "(T([32768, 30524], f32), T([64, 512], i64), T([], f32), S([64, 512, 30522]), S([32768, 30522]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_74: "f32[32768, 30524]", arg0_1: "i64[64, 512]", full_1: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        slice_tensor: "f32[32768, 30522]" = torch.ops.aten.slice.Tensor(addmm_74, 1, 0, -2);  addmm_74 = None
        view_default: "f32[64, 512, 30522]" = torch.ops.aten.view.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None
        constant_pad_nd_default: "i64[64, 513]" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1], -100.0);  arg0_1 = None
        slice_tensor_1: "i64[64, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, 1, 1, 9223372036854775807);  constant_pad_nd_default = None
        clone_default: "i64[64, 512]" = torch.ops.aten.clone.default(slice_tensor_1, memory_format = torch.contiguous_format);  slice_tensor_1 = None
        view_default_1: "f32[32768, 30522]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "i64[32768]" = torch.ops.aten.view.default(clone_default, [-1]);  clone_default = None
        amax_default: "f32[32768, 1]" = torch.ops.aten.amax.default(view_default_1, [1], True)
        sub_tensor: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(view_default_1, amax_default);  view_default_1 = amax_default = None
        exp_default: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[32768, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[32768]" = torch.ops.aten.ne.Scalar(view_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[32768]" = torch.ops.aten.where.self(ne_scalar, view_default_2, full_default);  view_default_2 = full_default = None
        unsqueeze_default: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[32768, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[32768]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[32768]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[32768]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_1);  neg_default = full_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None
        return div_tensor



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
