"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_infer_000
Pattern hash: 56d491110e60
Shape hash: 5cea262a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([192, 128, 128], f32), T([32, 6], f32), S([32, 6, 128, 128]), S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_2: "f32[192, 128, 128]", arg6_1: "f32[32, 6]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_0);  bmm_2 = _shape_param_0 = None
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        iota_default_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, 1);  iota_default_1 = None
        add_tensor: "i64[128, 1]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, 0);  unsqueeze_default_1 = None
        sub_tensor: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_default, add_tensor);  unsqueeze_default = add_tensor = None
        gt_scalar: "b8[128, 128]" = torch.ops.aten.gt.Scalar(sub_tensor, 0)
        convert_element_type_default: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(gt_scalar, torch.int64);  gt_scalar = None
        mul_tensor: "i64[128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 16);  convert_element_type_default = None
        add_tensor_1: "i64[128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, 0);  mul_tensor = None
        abs_default: "i64[128, 128]" = torch.ops.aten.abs.default(sub_tensor);  sub_tensor = None
        lt_scalar: "b8[128, 128]" = torch.ops.aten.lt.Scalar(abs_default, 8)
        convert_element_type_default_1: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(abs_default, torch.float32)
        div_tensor: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_default_1, 8);  convert_element_type_default_1 = None
        log_default: "f32[128, 128]" = torch.ops.aten.log.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_default, 2.772588722239781);  log_default = None
        mul_tensor_1: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_tensor_1, 8);  div_tensor_1 = None
        convert_element_type_default_2: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.int64);  mul_tensor_1 = None
        add_tensor_2: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, 8);  convert_element_type_default_2 = None
        full_default: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "i64[128, 128]" = torch.ops.aten.minimum.default(add_tensor_2, full_default);  add_tensor_2 = full_default = None
        where_self: "i64[128, 128]" = torch.ops.aten.where.self(lt_scalar, abs_default, minimum_default);  lt_scalar = abs_default = minimum_default = None
        add_tensor_3: "i64[128, 128]" = torch.ops.aten.add.Tensor(add_tensor_1, where_self);  add_tensor_1 = where_self = None
        embedding_default: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg6_1, add_tensor_3);  arg6_1 = add_tensor_3 = None
        permute_default: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_2: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_default, 0);  permute_default = None
        iota_default_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_4: "i64[128]" = torch.ops.aten.add.Tensor(iota_default_2, 0);  iota_default_2 = None
        unsqueeze_default_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor_4, 0);  add_tensor_4 = None
        unsqueeze_default_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        ge_scalar: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_default_5, 0);  unsqueeze_default_5 = None
        expand_default: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(ge_scalar, _shape_param_1);  ge_scalar = _shape_param_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  expand_default = full_default_1 = full_default_2 = None
        add_tensor_5: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_default_2, where_self_1);  unsqueeze_default_2 = where_self_1 = None
        add_tensor_6: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_default, add_tensor_5);  view_default = add_tensor_5 = None
        eq_scalar: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_6, -inf)
        logical_not_default: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_3: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_6, [-1], True)
        sub_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_6, amax_default);  add_tensor_6 = amax_default = None
        exp_default: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_2: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default_3, div_tensor_2);  logical_not_default_1 = full_default_3 = div_tensor_2 = None
        expand_default_1: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self_2, _shape_param_2);  where_self_2 = _shape_param_2 = None
        view_default_1: "f32[192, 128, 128]" = torch.ops.aten.view.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None
        return view_default_1

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
